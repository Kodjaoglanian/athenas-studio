//! ONNX Runtime backend (`ort`).
//!
//! Loads decoder (generative) and encoder (embedding) ONNX models with a
//! manual generation loop: HF `tokenizers` for (de)tokenization, minijinja
//! for the model's chat template, and KV-cache handling via the standard
//! `past_key_values.*` / `present.*` tensor names used by onnx-community
//! and onnxruntime-genai exports.
//!
//! Model path may be a single `.onnx` file or a directory containing the
//! model plus `tokenizer.json`/`genai_config.json`/`config.json`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
#[cfg(feature = "onnx-coreml")]
use ort::ep::CoreML;
#[cfg(feature = "onnx-directml")]
use ort::ep::DirectML;
#[cfg(feature = "onnx-rocm")]
use ort::ep::ROCm;
#[cfg(feature = "onnx-tensorrt")]
use ort::ep::TensorRT;
use ort::ep::CPU;
#[cfg(feature = "onnx-cuda")]
use ort::ep::CUDA;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::prelude::*;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use athenas_core::{AthenasError, GpuRuntime, HardwareInfo, Result};

use crate::backend::{Backend, ModelInfo};
use crate::types::{
    ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, InferenceStats, ModelLoadConfig, Role, StreamChunk,
    TokenizeRequest, TokenizeResponse,
};

/// True when this build can run ONNX models on a GPU (a GPU execution
/// provider was compiled in via a cargo feature). Without it every ONNX
/// load lands in RAM regardless of `gpu_layers` — pre-flight estimates
/// must treat the model as fully host-resident.
pub const GPU_OFFLOAD_CAPABLE: bool = cfg!(any(
    feature = "onnx-cuda",
    feature = "onnx-tensorrt",
    feature = "onnx-directml",
    feature = "onnx-coreml",
    feature = "onnx-rocm",
    feature = "onnx-openvino"
));

fn ort_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        ort::init().with_name("athenas").commit();
    });
}

/// Which generation/embedding layout the loaded model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxKind {
    /// Causal decoder: `input_ids`/`attention_mask`/`past_key_values.*` → `logits`/`present.*`
    Decoder,
    /// Encoder: `input_ids`/`attention_mask` → `last_hidden_state`/`sentence_embedding`
    Encoder,
}

/// A `past_key_values.*` input paired with the `present.*` output that feeds it.
struct KvFeed {
    input_name: String,
    output_name: String,
    dtype: TensorElementType,
    /// Concrete shape for the empty-past first run (seq dim already zeroed).
    empty_shape: Vec<i64>,
}

struct OnnxSession {
    session: Session,
    tokenizer: Tokenizer,
    kind: OnnxKind,
    model_name: String,
    context_size: u32,
    gpu_layers: i32,
    kv_feeds: Vec<KvFeed>,
    has_position_ids: bool,
    has_token_type_ids: bool,
    logits_name: Option<String>,
    /// Name of the encoder output to use for embeddings.
    embed_name: Option<String>,
    input_ids_i32: bool,
    eos_ids: Vec<i64>,
    pad_id: Option<i64>,
    bos_token: String,
    eos_token: String,
    chat_template: Option<String>,
}

/// Resolve the `.onnx` file and its directory from a file-or-directory path.
///
/// Priority inside a directory: `genai_config.json`'s decoder filename →
/// `model.onnx` / `decoder_model*.onnx` → the largest non-external `.onnx`.
fn resolve_onnx_path(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let err = |m: String| AthenasError::Backend(m);
    if path.is_file() {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
        {
            let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            return Ok((path.to_path_buf(), dir));
        }
        return Err(err(format!("not an .onnx file: {}", path.display())));
    }
    if !path.is_dir() {
        return Err(err(format!("model path not found: {}", path.display())));
    }

    // genai_config.json names the decoder file explicitly.
    let genai_cfg = path.join("genai_config.json");
    if genai_cfg.is_file() {
        if let Some(v) = std::fs::read_to_string(&genai_cfg)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        {
            for key in ["model.decoder.filename", "model.filename"] {
                let mut node = &v;
                for part in key.split('.') {
                    node = &node[part];
                }
                if let Some(f) = node.as_str() {
                    let p = path.join(f);
                    if p.is_file() {
                        return Ok((p, path.to_path_buf()));
                    }
                }
            }
        }
    }

    let mut onnx_files: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(path)
        .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?
        .flatten()
    {
        let p = e.path();
        if p.extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("onnx"))
        {
            onnx_files.push(p);
        }
    }
    onnx_files.sort();
    if onnx_files.is_empty() {
        return Err(err(format!("no .onnx file in {}", path.display())));
    }
    for name in [
        "model.onnx",
        "decoder_model_merged.onnx",
        "decoder_model.onnx",
    ] {
        if let Some(p) = onnx_files
            .iter()
            .find(|p| p.file_name().is_some_and(|n| n == name))
        {
            return Ok((p.clone(), path.to_path_buf()));
        }
    }
    // Otherwise pick the largest .onnx (weights are usually inline or in
    // a sibling `.onnx_data`, which the runtime resolves relative to the model).
    onnx_files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    let p = onnx_files.last().cloned().unwrap();
    Ok((p, path.to_path_buf()))
}

fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    std::fs::read_to_string(dir.join(name))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Token ids from a JSON value that may be an int or an array of ints.
fn json_ids(v: &serde_json::Value) -> Vec<i64> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().into_iter().collect(),
        serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_i64()).collect(),
        _ => Vec::new(),
    }
}

/// Extract declared tensor shape + dtype from an input/output descriptor.
/// Dynamic dimensions are `-1` in the declared shape.
fn tensor_meta(vt: &ValueType) -> (Option<TensorElementType>, Vec<i64>) {
    match vt {
        ValueType::Tensor { ty, shape, .. } => (Some(*ty), shape.iter().copied().collect()),
        _ => (None, Vec::new()),
    }
}

fn empty_kv_tensor(dtype: TensorElementType, shape: &[i64]) -> Result<DynValue> {
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    match dtype {
        TensorElementType::Float32 => Tensor::<f32>::from_array((dims, Vec::new()))
            .map(|t| t.into_dyn())
            .map_err(|e| AthenasError::Backend(e.to_string())),
        TensorElementType::Float16 => Tensor::<half::f16>::from_array((dims, Vec::new()))
            .map(|t| t.into_dyn())
            .map_err(|e| AthenasError::Backend(e.to_string())),
        other => Err(AthenasError::Backend(format!(
            "unsupported KV cache dtype {other:?} — only float32/float16 past_key_values are supported"
        ))),
    }
}

/// Sample one token from a logits slice with temperature + top-k + top-p.
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut impl Rng,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    // Greedy when temperature ≈ 0.
    if temperature <= 1e-5 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    // (id, scaled logit) sorted desc.
    let mut vals: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l / temperature))
        .collect();
    vals.sort_by(|a, b| b.1.total_cmp(&a.1));
    if top_k > 0 && vals.len() > top_k {
        vals.truncate(top_k);
    }
    // Softmax.
    let max = vals[0].1;
    let mut probs: Vec<f32> = vals.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return vals[0].0;
    }
    for p in &mut probs {
        *p /= sum;
    }
    // Nucleus (top-p).
    if top_p > 0.0 && top_p < 1.0 {
        let mut acc = 0.0;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if acc >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        vals.truncate(cutoff);
        probs.truncate(cutoff);
    }
    match WeightedIndex::new(&probs) {
        Ok(dist) => vals[dist.sample(rng)].0,
        Err(_) => vals[0].0,
    }
}

/// Render the chat prompt with the model's Jinja template, or fall back
/// to a simple ChatML-ish format.
fn render_chat_prompt(s: &OnnxSession, request: &ChatRequest) -> Result<String> {
    let msgs: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role.to_string(),
                "content": m.content.as_text(),
            })
        })
        .collect();
    if let Some(tpl) = &s.chat_template {
        let env = minijinja::Environment::new();
        let ctx = minijinja::context! {
            messages => msgs,
            add_generation_prompt => true,
            bos_token => s.bos_token.clone(),
            eos_token => s.eos_token.clone(),
        };
        match env.render_str(tpl, &ctx) {
            Ok(t) => return Ok(t),
            Err(e) => {
                warn!("chat template render failed ({e}); using fallback format");
            }
        }
    }
    // Fallback: system + turns in a minimal instruct format.
    let mut out = String::new();
    for m in &request.messages {
        let text = m.content.as_text();
        match m.role {
            Role::System => {
                out.push_str(&format!("<|im_start|>system\n{text}<|im_end|>\n"));
            }
            Role::User => {
                out.push_str(&format!("<|im_start|>user\n{text}<|im_end|>\n"));
            }
            Role::Assistant => {
                out.push_str(&format!("<|im_start|>assistant\n{text}<|im_end|>\n"));
            }
            Role::Tool => {
                out.push_str(&format!("<|im_start|>tool\n{text}<|im_end|>\n"));
            }
        }
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

/// Build inputs for one forward pass. `past` holds the owned KV values from
/// the previous step (empty vec on the first pass → empty tensors are fed).
#[allow(clippy::too_many_arguments)]
fn run_step<'s>(
    session: &'s mut Session,
    input_ids_i32: bool,
    has_position_ids: bool,
    kv_feeds: &[KvFeed],
    ids: &[i64],
    attention_len: usize,
    position_offset: usize,
    past: Vec<(String, DynValue)>,
) -> Result<SessionOutputs<'s>> {
    let n = ids.len();
    let ids_t = if input_ids_i32 {
        let v: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        Tensor::from_array((vec![1usize, n], v))
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .into_dyn()
    } else {
        Tensor::from_array((vec![1usize, n], ids.to_vec()))
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .into_dyn()
    };
    let mask_t = Tensor::from_array((vec![1usize, attention_len], vec![1i64; attention_len]))
        .map_err(|e| AthenasError::Backend(e.to_string()))?
        .into_dyn();

    let mut inputs: Vec<(String, DynValue)> = vec![
        ("input_ids".to_string(), ids_t),
        ("attention_mask".to_string(), mask_t),
    ];
    if has_position_ids {
        let pos: Vec<i64> = (position_offset..position_offset + n)
            .map(|x| x as i64)
            .collect();
        let pos_t = Tensor::from_array((vec![1usize, n], pos))
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .into_dyn();
        inputs.push(("position_ids".to_string(), pos_t));
    }
    if past.is_empty() {
        for kv in kv_feeds {
            inputs.push((
                kv.input_name.clone(),
                empty_kv_tensor(kv.dtype, &kv.empty_shape)?,
            ));
        }
    } else {
        inputs.extend(past);
    }
    session
        .run(inputs)
        .map_err(|e| AthenasError::Backend(format!("ONNX run failed: {e}")))
}

/// Pull owned `present.*` values out of outputs, ordered to match kv_feeds.
fn take_past(outputs: &mut SessionOutputs<'_>, kv_feeds: &[KvFeed]) -> Vec<(String, DynValue)> {
    kv_feeds
        .iter()
        .filter_map(|kv| {
            outputs
                .remove(kv.output_name.as_str())
                .map(|v| (kv.input_name.clone(), v))
        })
        .collect()
}

/// Last-position logits as a Vec<f32>.
fn take_logits(name: &str, outputs: &SessionOutputs<'_>) -> Result<Vec<f32>> {
    let v = outputs
        .get(name)
        .ok_or_else(|| AthenasError::Backend(format!("no '{name}' output in model")))?;
    let (shape, data) = v
        .try_extract_tensor::<f32>()
        .map_err(|e| AthenasError::Backend(format!("cannot read logits: {e}")))?;
    let dims: Vec<i64> = shape.iter().copied().collect();
    let vocab = *dims.last().unwrap_or(&0) as usize;
    if vocab == 0 || data.len() % vocab != 0 {
        return Err(AthenasError::Backend(format!(
            "unexpected logits shape {dims:?}"
        )));
    }
    Ok(data[data.len() - vocab..].to_vec())
}

impl OnnxSession {
    /// Full generation loop for a tokenized prompt. `on_text` is called with
    /// decoded text deltas for streaming.
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        max_tokens: u32,
        seed: Option<u64>,
        stop: &[String],
        mut on_text: impl FnMut(&str),
    ) -> Result<(String, InferenceStats)> {
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| AthenasError::Backend(format!("tokenize failed: {e}")))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&t| t as i64).collect();
        let prompt_tokens = ids.len() as u32;
        if prompt_tokens >= self.context_size {
            return Err(AthenasError::Backend(format!(
                "prompt ({} tokens) exceeds context size {}",
                prompt_tokens, self.context_size
            )));
        }
        let budget = max_tokens
            .min(self.context_size.saturating_sub(prompt_tokens))
            .max(1);

        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
        };

        let t0 = std::time::Instant::now();
        let mut generated: Vec<i64> = Vec::new();
        let mut text = String::new();
        let prompt_len = ids.len();

        // Prefill: all prompt tokens at once, empty KV cache.
        let mut outputs = run_step(
            &mut self.session,
            self.input_ids_i32,
            self.has_position_ids,
            &self.kv_feeds,
            &ids,
            prompt_len,
            0,
            Vec::new(),
        )?;
        let mut past = take_past(&mut outputs, &self.kv_feeds);
        let logits_name = self.logits_name.clone().unwrap_or_else(|| "logits".into());
        let mut logits = take_logits(&logits_name, &outputs)?;
        drop(outputs);

        for total_len in (prompt_len..).take(budget as usize) {
            let tok = sample_token(&logits, temperature, 0, top_p, &mut rng) as i64;
            generated.push(tok);
            let mut finished =
                self.eos_ids.contains(&tok) || (generated.len() > 1 && self.pad_id == Some(tok));

            // Decode cumulative and emit the delta (handles BPE merges).
            let gen_ids: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
            let new_text = self.tokenizer.decode(&gen_ids, true).unwrap_or_default();
            // Earliest stop string truncates the output.
            let mut cut: Option<usize> = None;
            for s in stop {
                if let Some(p) = new_text.find(s.as_str()) {
                    cut = Some(cut.map_or(p, |q: usize| q.min(p)));
                }
            }
            let emit_to = cut.unwrap_or(new_text.len());
            if emit_to > text.len() {
                on_text(&new_text[text.len()..emit_to]);
            }
            if cut.is_some() {
                finished = true;
                text = new_text[..emit_to].to_string();
            } else {
                text = new_text;
            }
            if finished {
                break;
            }

            // Decode step: single token, grown attention mask.
            let step = [tok];
            let mut out = run_step(
                &mut self.session,
                self.input_ids_i32,
                self.has_position_ids,
                &self.kv_feeds,
                &step,
                total_len + 1,
                total_len,
                past,
            )?;
            past = take_past(&mut out, &self.kv_feeds);
            logits = take_logits(&logits_name, &out)?;
        }

        let elapsed = t0.elapsed().as_millis() as u64;
        let n = generated.len() as u32;
        Ok((
            text,
            InferenceStats {
                tokens_generated: n,
                tokens_prompt: prompt_tokens,
                time_total_ms: elapsed,
                tokens_per_second: if elapsed > 0 {
                    n as f32 / (elapsed as f32 / 1000.0)
                } else {
                    0.0
                },
            },
        ))
    }

    /// Mean-pooled (mask-aware) embedding for one input.
    fn embed_one(&mut self, text: &str) -> Result<(Vec<f32>, u32)> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AthenasError::Backend(format!("tokenize failed: {e}")))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&t| t as i64).collect();
        let n = ids.len();
        let ntokens = n as u32;

        let ids_t = if self.input_ids_i32 {
            Tensor::from_array((
                vec![1usize, n],
                ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            ))
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .into_dyn()
        } else {
            Tensor::from_array((vec![1usize, n], ids))
                .map_err(|e| AthenasError::Backend(e.to_string()))?
                .into_dyn()
        };
        let mask_t = Tensor::from_array((vec![1usize, n], vec![1i64; n]))
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .into_dyn();
        let mut inputs: Vec<(String, DynValue)> = vec![
            ("input_ids".to_string(), ids_t),
            ("attention_mask".to_string(), mask_t),
        ];
        if self.has_token_type_ids {
            let tt = Tensor::from_array((vec![1usize, n], vec![0i64; n]))
                .map_err(|e| AthenasError::Backend(e.to_string()))?
                .into_dyn();
            inputs.push(("token_type_ids".to_string(), tt));
        }
        let mut outputs = self
            .session
            .run(inputs)
            .map_err(|e| AthenasError::Backend(format!("ONNX run failed: {e}")))?;

        // Prefer a dedicated sentence/embedding output; else mean-pool the
        // first tensor output over the sequence dim.
        let name = self
            .embed_name
            .clone()
            .unwrap_or_else(|| outputs.keys().next().unwrap_or_default().to_string());
        let v = outputs
            .remove(name.as_str())
            .ok_or_else(|| AthenasError::Backend(format!("no '{name}' output")))?;
        let (shape, data) = v
            .try_extract_tensor::<f32>()
            .map_err(|e| AthenasError::Backend(format!("cannot read embedding: {e}")))?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let emb = match dims.len() {
            // [1, seq, hidden] → mean pool
            3 => {
                let hidden = dims[2];
                let mut pooled = vec![0f32; hidden];
                for t in 0..dims[1] {
                    for h in 0..hidden {
                        pooled[h] += data[t * hidden + h];
                    }
                }
                for p in &mut pooled {
                    *p /= dims[1].max(1) as f32;
                }
                pooled
            }
            // [1, hidden] or [hidden]
            2 => data[..dims[1]].to_vec(),
            1 => data.to_vec(),
            _ => {
                return Err(AthenasError::Backend(format!(
                    "unexpected embedding output shape {dims:?}"
                )))
            }
        };
        // L2-normalize (standard for embedding models).
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb = if norm > 0.0 {
            emb.iter().map(|x| x / norm).collect()
        } else {
            emb
        };
        Ok((emb, ntokens))
    }
}

/// ONNX Runtime backend. Shared via `Arc<Mutex>` so `boxed_clone()` handles
/// operate on the same loaded session.
pub struct OnnxBackend {
    inner: Arc<Mutex<Option<OnnxSession>>>,
    loaded: Arc<AtomicBool>,
    hardware: HardwareInfo,
}

impl OnnxBackend {
    pub fn new(hardware: &HardwareInfo) -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            loaded: Arc::new(AtomicBool::new(false)),
            hardware: hardware.clone(),
        }
    }

    // ort's builder methods return a large Err type carrying the consumed
    // builder — nothing we can do about the upstream signature.
    #[allow(clippy::result_large_err)]
    fn load(&self, config: &ModelLoadConfig) -> Result<()> {
        let (model_file, dir) = resolve_onnx_path(Path::new(&config.model_path))?;
        info!("loading ONNX model: {}", model_file.display());

        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(AthenasError::Backend(format!(
                "tokenizer.json not found in {} — required for the ONNX backend",
                dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AthenasError::Backend(format!("failed to load tokenizer: {e}")))?;

        // Model metadata: genai_config.json > config.json > tokenizer_config.json.
        let genai = read_json(&dir, "genai_config.json");
        let cfg = read_json(&dir, "config.json");
        let tok_cfg = read_json(&dir, "tokenizer_config.json");

        let mut eos_ids: Vec<i64> = Vec::new();
        for v in [&genai, &cfg, &tok_cfg].into_iter().flatten() {
            for key in ["eos_token_id", "model.eos_token_id"] {
                let mut node = v;
                for part in key.split('.') {
                    node = &node[part];
                }
                eos_ids.extend(json_ids(node));
            }
        }
        eos_ids.sort_unstable();
        eos_ids.dedup();
        // Also treat the tokenizer's eos string as a stop id.
        let eos_token = tok_cfg
            .as_ref()
            .and_then(|v| v["eos_token"].as_str().map(String::from))
            .unwrap_or_default();
        if let Some(id) = tokenizer.token_to_id(&eos_token) {
            eos_ids.push(id as i64);
        }
        let pad_id = cfg
            .as_ref()
            .and_then(|v| v["pad_token_id"].as_i64())
            .or_else(|| {
                genai
                    .as_ref()
                    .and_then(|v| v["model"]["pad_token_id"].as_i64())
            });
        let bos_token = tok_cfg
            .as_ref()
            .and_then(|v| v["bos_token"].as_str().map(String::from))
            .unwrap_or_default();

        let chat_template = tok_cfg.as_ref().and_then(|v| {
            let ct = &v["chat_template"];
            if let Some(s) = ct.as_str() {
                Some(s.to_string())
            } else if let Some(arr) = ct.as_array() {
                arr.iter()
                    .find(|x| x["name"].as_str() == Some("default"))
                    .or_else(|| arr.first())
                    .and_then(|x| x["template"].as_str().map(String::from))
            } else {
                None
            }
        });

        // Execution providers. GPU EPs only exist when the matching cargo
        // feature was enabled at build time (each picks a different ORT
        // binary dist); otherwise we warn and fall back to CPU.
        let gpu_ok = config.gpu_layers != 0 && config.gpu_runtime != GpuRuntime::Cpu;
        let eps = {
            let mut v: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
            if gpu_ok {
                let want_cuda = matches!(config.gpu_runtime, GpuRuntime::Cuda)
                    || (config.gpu_runtime == GpuRuntime::Auto && self.hardware.has_cuda);
                let want_dml = cfg!(windows)
                    && (matches!(config.gpu_runtime, GpuRuntime::Auto | GpuRuntime::Vulkan)
                        && !self.hardware.has_cuda);
                let want_coreml = cfg!(target_os = "macos")
                    && matches!(config.gpu_runtime, GpuRuntime::Auto | GpuRuntime::Metal);
                let want_rocm = matches!(config.gpu_runtime, GpuRuntime::Rocm)
                    || (config.gpu_runtime == GpuRuntime::Auto && self.hardware.has_rocm);

                #[cfg(feature = "onnx-cuda")]
                if want_cuda {
                    v.push(
                        CUDA::default()
                            .with_device_id(config.gpu_device.unwrap_or(0) as i32)
                            .build(),
                    );
                }
                #[cfg(feature = "onnx-tensorrt")]
                if want_cuda {
                    // TensorRT first when built — CUDA EP remains as fallback.
                    v.insert(
                        0,
                        TensorRT::default()
                            .with_device_id(config.gpu_device.unwrap_or(0) as i32)
                            .build(),
                    );
                }
                #[cfg(feature = "onnx-directml")]
                if want_dml {
                    v.push(
                        DirectML::default()
                            .with_device_id(config.gpu_device.unwrap_or(0) as i32)
                            .build(),
                    );
                }
                #[cfg(feature = "onnx-coreml")]
                if want_coreml {
                    v.push(CoreML::default().build());
                }
                #[cfg(feature = "onnx-rocm")]
                if want_rocm {
                    v.push(
                        ROCm::default()
                            .with_device_id(config.gpu_device.unwrap_or(0) as i32)
                            .build(),
                    );
                }

                let requested = want_cuda || want_dml || want_coreml || want_rocm;
                if requested && v.is_empty() {
                    warn!(
                        "GPU requested for ONNX model but this build has no GPU execution \
                         providers compiled in — using CPU. Rebuild with an onnx-* feature \
                         (e.g. --features onnx-cuda) to enable GPU offload."
                    );
                }
            }
            v.push(CPU::default().build());
            v
        };

        let session = Session::builder()
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .and_then(|b| b.with_intra_threads(config.threads.max(1) as usize))
            .and_then(|b| b.with_execution_providers(eps))
            .map_err(|e| AthenasError::Backend(e.to_string()))?
            .commit_from_file(&model_file)
            .map_err(|e| {
                AthenasError::Backend(format!("failed to load {}: {e}", model_file.display()))
            })?;

        // Inspect declared inputs/outputs to pick a layout.
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        debug!("onnx inputs: {input_names:?} outputs: {output_names:?}");

        let input_ids_i32 = session
            .inputs()
            .iter()
            .find(|i| i.name() == "input_ids")
            .map(|i| matches!(tensor_meta(i.dtype()).0, Some(TensorElementType::Int32)))
            .unwrap_or(false);

        let logits_name = output_names
            .iter()
            .find(|n| n.as_str() == "logits" || n.contains("logits"))
            .cloned();
        let past_inputs: Vec<&String> = input_names
            .iter()
            .filter(|n| n.starts_with("past_key_values") || n.starts_with("past."))
            .collect();
        // A decoder needs logits AND KV-cache inputs — a bare `logits` output
        // with no past_key_values is a classifier/encoder, not a generator.
        let kind = if logits_name.is_some() && !past_inputs.is_empty() {
            OnnxKind::Decoder
        } else if input_names.iter().any(|n| n == "input_ids") {
            OnnxKind::Encoder
        } else {
            return Err(AthenasError::Backend(format!(
                "unsupported ONNX model layout: inputs {input_names:?} — expected input_ids"
            )));
        };

        // KV feeds: pair `past_key_values.*` inputs with `present.*` outputs.
        // Both enumerations are ordered (layer-major, key before value), so
        // positional pairing is safe; fall back to name mapping first.
        let present_outputs: Vec<&String> = output_names
            .iter()
            .filter(|n| n.starts_with("present"))
            .collect();
        let mut kv_feeds = Vec::new();
        for (idx, pin) in past_inputs.iter().enumerate() {
            let out_name = {
                // genai style: past_key_values.N.key → present.N.key
                let cand = pin.replacen("past_key_values", "present", 1);
                let cand2 = pin.replacen("past", "present", 1);
                if output_names.contains(&cand) {
                    cand
                } else if output_names.contains(&cand2) {
                    cand2
                } else {
                    present_outputs
                        .get(idx)
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                }
            };
            if out_name.is_empty() {
                return Err(AthenasError::Backend(format!(
                    "no matching present.* output for input {pin}"
                )));
            }
            let (dtype, shape) = session
                .inputs()
                .iter()
                .find(|i| i.name() == pin.as_str())
                .map(|i| tensor_meta(i.dtype()))
                .unwrap_or((None, Vec::new()));
            let dtype = dtype.unwrap_or(TensorElementType::Float32);
            // Empty-past shape: batch dim → 1, dynamic dims (-1) → 0 (the
            // past seq dim is dynamic so the tensor ends up empty), keep
            // concrete dims like num_heads/head_dim.
            let empty_shape: Vec<i64> = shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if i == 0 && d != 0 { 1 } else { d.max(0) })
                .collect();
            kv_feeds.push(KvFeed {
                input_name: pin.to_string(),
                output_name: out_name,
                dtype,
                empty_shape,
            });
        }

        let embed_name = output_names
            .iter()
            .find(|n| {
                let n = n.as_str();
                n == "sentence_embedding" || n == "embeddings" || n == "last_hidden_state"
            })
            .cloned();

        let model_name = Path::new(&config.model_path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "onnx-model".to_string());

        let sess = OnnxSession {
            session,
            tokenizer,
            kind,
            model_name,
            context_size: config.context_size,
            gpu_layers: config.gpu_layers,
            kv_feeds,
            has_position_ids: input_names.iter().any(|n| n == "position_ids"),
            has_token_type_ids: input_names.iter().any(|n| n == "token_type_ids"),
            logits_name,
            embed_name,
            input_ids_i32,
            eos_ids,
            pad_id,
            bos_token,
            eos_token,
            chat_template,
        };
        *self.inner.lock().expect("onnx mutex poisoned") = Some(sess);
        self.loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn with_session<R>(&self, f: impl FnOnce(&mut OnnxSession) -> Result<R>) -> Result<R> {
        let mut g = self.inner.lock().expect("onnx mutex poisoned");
        let s = g
            .as_mut()
            .ok_or_else(|| AthenasError::Backend("no model loaded".into()))?;
        f(s)
    }
}

#[async_trait]
impl Backend for OnnxBackend {
    fn name(&self) -> &str {
        "onnx"
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn load_model(&mut self, config: ModelLoadConfig) -> Result<()> {
        ort_env();
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || this.load(&config))
            .await
            .map_err(|e| AthenasError::Backend(format!("load task failed: {e}")))?
    }

    async fn unload_model(&mut self) -> Result<()> {
        *self.inner.lock().expect("onnx mutex poisoned") = None;
        self.loaded.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || {
            this.with_session(|s| {
                if s.kind != OnnxKind::Decoder {
                    return Err(AthenasError::Backend(
                        "loaded ONNX model is not a decoder — use /v1/embeddings".into(),
                    ));
                }
                let prompt = render_chat_prompt(s, &request)?;
                let (text, stats) = s.generate(
                    &prompt,
                    request.temperature.unwrap_or(0.7),
                    request.top_p.unwrap_or(0.9),
                    request.max_tokens.unwrap_or(2048),
                    request.seed,
                    request.stop.as_deref().unwrap_or(&[]),
                    |_| {},
                )?;
                Ok(ChatResponse {
                    model: request.model.clone(),
                    message: crate::types::ChatMessage::assistant(text),
                    stats,
                    tool_calls: None,
                    finish_reason: Some("stop".to_string()),
                })
            })
        })
        .await
        .map_err(|e| AthenasError::Backend(format!("chat task failed: {e}")))?
    }

    async fn chat_stream(&self, request: ChatRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()> {
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || {
            let r = this.with_session(|s| {
                if s.kind != OnnxKind::Decoder {
                    return Err(AthenasError::Backend(
                        "loaded ONNX model is not a decoder".into(),
                    ));
                }
                let prompt = render_chat_prompt(s, &request)?;
                let tx2 = tx.clone();
                let (text, stats) = s.generate(
                    &prompt,
                    request.temperature.unwrap_or(0.7),
                    request.top_p.unwrap_or(0.9),
                    request.max_tokens.unwrap_or(2048),
                    request.seed,
                    request.stop.as_deref().unwrap_or(&[]),
                    |delta| {
                        let _ = tx2.blocking_send(StreamChunk {
                            text: delta.to_string(),
                            done: false,
                            stats: None,
                            is_reasoning: false,
                        });
                    },
                )?;
                let _ = tx.blocking_send(StreamChunk {
                    text,
                    done: true,
                    stats: Some(stats),
                    is_reasoning: false,
                });
                Ok(())
            });
            if let Err(e) = &r {
                let _ = tx.blocking_send(StreamChunk {
                    text: String::new(),
                    done: true,
                    stats: None,
                    is_reasoning: false,
                });
                warn!("onnx chat_stream failed: {e}");
            }
            r
        })
        .await
        .map_err(|e| AthenasError::Backend(format!("stream task failed: {e}")))?
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || {
            this.with_session(|s| {
                if s.kind != OnnxKind::Decoder {
                    return Err(AthenasError::Backend(
                        "loaded ONNX model is not a decoder".into(),
                    ));
                }
                let (text, stats) = s.generate(
                    &request.prompt,
                    request.temperature.unwrap_or(0.7),
                    request.top_p.unwrap_or(0.9),
                    request.max_tokens.unwrap_or(2048),
                    request.seed,
                    request.stop.as_deref().unwrap_or(&[]),
                    |_| {},
                )?;
                Ok(CompletionResponse {
                    model: request.model.clone(),
                    text,
                    stats,
                })
            })
        })
        .await
        .map_err(|e| AthenasError::Backend(format!("complete task failed: {e}")))?
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || {
            this.with_session(|s| {
                if s.kind != OnnxKind::Decoder {
                    return Err(AthenasError::Backend(
                        "loaded ONNX model is not a decoder".into(),
                    ));
                }
                let tx2 = tx.clone();
                let (text, stats) = s.generate(
                    &request.prompt,
                    request.temperature.unwrap_or(0.7),
                    request.top_p.unwrap_or(0.9),
                    request.max_tokens.unwrap_or(2048),
                    request.seed,
                    request.stop.as_deref().unwrap_or(&[]),
                    |delta| {
                        let _ = tx2.blocking_send(StreamChunk {
                            text: delta.to_string(),
                            done: false,
                            stats: None,
                            is_reasoning: false,
                        });
                    },
                )?;
                let _ = tx.blocking_send(StreamChunk {
                    text,
                    done: true,
                    stats: Some(stats),
                    is_reasoning: false,
                });
                Ok(())
            })
        })
        .await
        .map_err(|e| AthenasError::Backend(format!("stream task failed: {e}")))?
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || {
            this.with_session(|s| {
                let texts = match &request.input {
                    EmbeddingInput::Single(t) => vec![t.clone()],
                    EmbeddingInput::Batch(v) => v.clone(),
                };
                let mut data = Vec::with_capacity(texts.len());
                let mut total = 0u32;
                for (i, t) in texts.iter().enumerate() {
                    let (emb, nt) = s.embed_one(t)?;
                    total += nt;
                    data.push(crate::types::EmbeddingData {
                        object: "embedding".to_string(),
                        embedding: emb,
                        index: i,
                    });
                }
                Ok(EmbeddingResponse {
                    object: "list".to_string(),
                    data,
                    model: request.model.clone(),
                    usage: crate::types::EmbeddingUsage {
                        prompt_tokens: total,
                        total_tokens: total,
                    },
                })
            })
        })
        .await
        .map_err(|e| AthenasError::Backend(format!("embed task failed: {e}")))?
    }

    async fn tokenize(&self, request: TokenizeRequest) -> Result<TokenizeResponse> {
        let this = self.clone_shared();
        tokio::task::spawn_blocking(move || {
            this.with_session(|s| {
                if request.detokenize {
                    let ids: Vec<u32> = request
                        .text
                        .split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    let text = s
                        .tokenizer
                        .decode(&ids, true)
                        .map_err(|e| AthenasError::Backend(format!("decode failed: {e}")))?;
                    Ok(TokenizeResponse {
                        tokens: ids,
                        count: text.len(),
                        text: Some(text),
                    })
                } else {
                    let enc = s
                        .tokenizer
                        .encode(request.text.as_str(), true)
                        .map_err(|e| AthenasError::Backend(format!("tokenize failed: {e}")))?;
                    let tokens = enc.get_ids().to_vec();
                    Ok(TokenizeResponse {
                        count: tokens.len(),
                        tokens,
                        text: None,
                    })
                }
            })
        })
        .await
        .map_err(|e| AthenasError::Backend(format!("tokenize task failed: {e}")))?
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let g = self.inner.lock().ok()?;
        g.as_ref().map(|s| ModelInfo {
            name: s.model_name.clone(),
            context_size: s.context_size,
            gpu_layers: s.gpu_layers,
            backend_name: "onnx".to_string(),
        })
    }

    fn boxed_clone(&self) -> Box<dyn Backend> {
        Box::new(self.clone_shared())
    }
}

impl OnnxBackend {
    fn clone_shared(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            loaded: Arc::clone(&self.loaded),
            hardware: self.hardware.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_onnx_path_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("m.onnx");
        fs::write(&f, b"x").unwrap();
        let (mf, d) = resolve_onnx_path(&f).unwrap();
        assert_eq!(mf, f);
        assert_eq!(d, dir.path());
    }

    #[test]
    fn resolve_onnx_path_dir_prefers_model_onnx() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.onnx"), b"xx").unwrap();
        fs::write(dir.path().join("model.onnx"), b"x").unwrap();
        fs::write(dir.path().join("model.onnx_data"), b"big").unwrap();
        let (mf, _) = resolve_onnx_path(dir.path()).unwrap();
        assert_eq!(mf.file_name().unwrap(), "model.onnx");
    }

    #[test]
    fn resolve_onnx_path_genai_config() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("custom_quant.onnx"), b"x").unwrap();
        fs::write(
            dir.path().join("genai_config.json"),
            r#"{"model":{"decoder":{"filename":"custom_quant.onnx"}}}"#,
        )
        .unwrap();
        let (mf, _) = resolve_onnx_path(dir.path()).unwrap();
        assert_eq!(mf.file_name().unwrap(), "custom_quant.onnx");
    }

    #[test]
    fn resolve_onnx_path_missing() {
        assert!(resolve_onnx_path(Path::new("/nonexistent")).is_err());
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_onnx_path(dir.path()).is_err());
    }

    #[test]
    fn sample_token_greedy() {
        let logits = [0.1f32, 5.0, 1.0, 2.0];
        assert_eq!(
            sample_token(
                &logits,
                0.0,
                0,
                1.0,
                &mut rand::rngs::StdRng::seed_from_u64(1)
            ),
            1
        );
    }

    #[test]
    fn sample_token_top_k() {
        let logits = [10.0f32, 5.0, 1.0];
        // top_k=1 → always argmax even with temperature
        for _ in 0..10 {
            assert_eq!(sample_token(&logits, 1.0, 1, 1.0, &mut rand::rng()), 0);
        }
    }

    #[test]
    fn sample_token_seeded_deterministic() {
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let a = sample_token(
            &logits,
            0.8,
            0,
            0.9,
            &mut rand::rngs::StdRng::seed_from_u64(42),
        );
        let b = sample_token(
            &logits,
            0.8,
            0,
            0.9,
            &mut rand::rngs::StdRng::seed_from_u64(42),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn json_ids_scalar_and_array() {
        assert_eq!(json_ids(&serde_json::json!(128001)), vec![128001]);
        assert_eq!(
            json_ids(&serde_json::json!([128001, 128008])),
            vec![128001, 128008]
        );
        assert!(json_ids(&serde_json::json!("x")).is_empty());
    }
}
