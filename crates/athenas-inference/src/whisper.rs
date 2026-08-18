use std::path::PathBuf;
use tracing::info;

use athenas_core::{AthenasError, Result};

use crate::types::{TranscriptionRequest, TranscriptionResponse, TranscriptionSegment};

/// Whisper backend — wraps the whisper-cli binary for audio transcription.
///
/// whisper.cpp is a separate binary from llama-server. It uses GGUF models
/// with the "whisper" architecture (e.g. whisper-large-v3-Q4_K_M.gguf).
/// The binary is auto-downloaded to ~/.athenas/bin/whisper-cli on first use.
pub struct WhisperBackend {
    /// Path to the whisper-cli binary
    cli_path: PathBuf,
}

impl WhisperBackend {
    /// Create a new WhisperBackend, auto-downloading whisper-cli if needed.
    pub async fn new() -> Result<Self> {
        let cli_path = crate::backend_setup::ensure_whisper_cli().await?;
        Ok(Self { cli_path })
    }

    /// Transcribe an audio file using whisper-cli.
    ///
    /// The audio file must exist on disk (whisper-cli reads from a file path).
    /// Returns the transcribed text and optional segments with timestamps.
    pub async fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResponse> {
        // Write audio data to a temp file
        let tmp_dir = std::env::temp_dir();
        let tmp_audio = tmp_dir.join(format!("athenas-whisper-{}", request.filename));
        std::fs::write(&tmp_audio, &request.audio_data).map_err(|e| {
            AthenasError::Backend(format!("Failed to write temp audio file: {}", e))
        })?;

        // Determine output format
        let format = request.response_format.as_deref().unwrap_or("json");
        let output_file = tmp_audio.with_extension(format!("{}.txt", format));

        // Build whisper-cli command
        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("--model")
            .arg(&request.model)
            .arg("--file")
            .arg(&tmp_audio)
            .arg("--output-file")
            .arg(&output_file)
            .arg("--output-format")
            .arg(format)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Language
        if let Some(ref lang) = request.language {
            if lang != "auto" {
                cmd.arg("--language").arg(lang);
            }
        }

        // Translate to English
        if request.translate.unwrap_or(false) {
            cmd.arg("--translate");
        }

        // Max segment length
        if let Some(max_len) = request.max_len {
            cmd.arg("--max-len").arg(max_len.to_string());
        }

        // Set LD_LIBRARY_PATH on Unix so it finds shared libs
        #[cfg(unix)]
        if let Some(parent) = self.cli_path.parent() {
            let lib_path = parent.to_string_lossy().to_string();
            let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let new_ld_path = if existing.is_empty() {
                lib_path
            } else {
                format!("{}:{}", lib_path, existing)
            };
            cmd.env("LD_LIBRARY_PATH", new_ld_path);
        }

        // Prevent console window on Windows
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        info!(
            "Running whisper-cli on {} with model {}",
            request.filename, request.model
        );

        let output = cmd
            .output()
            .await
            .map_err(|e| AthenasError::Backend(format!("Failed to run whisper-cli: {}", e)))?;

        // Clean up temp audio file
        let _ = std::fs::remove_file(&tmp_audio);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Clean up output file if it exists
            let _ = std::fs::remove_file(&output_file);
            return Err(AthenasError::Backend(format!(
                "whisper-cli failed (exit code: {:?})\nstdout: {}\nstderr: {}",
                output.status.code(),
                stdout,
                stderr
            )));
        }

        // Read the output file
        let output_content = std::fs::read_to_string(&output_file)
            .map_err(|e| AthenasError::Backend(format!("Failed to read whisper output: {}", e)))?;

        // Clean up output file
        let _ = std::fs::remove_file(&output_file);

        // Parse output based on format
        let response = match format {
            "json" => self.parse_json_output(&output_content)?,
            "srt" | "vtt" => Self::parse_subtitle_output(&output_content, format),
            _ => TranscriptionResponse {
                text: output_content.trim().to_string(),
                language: None,
                duration: None,
                segments: Vec::new(),
            },
        };

        Ok(response)
    }

    /// Parse JSON output from whisper-cli.
    fn parse_json_output(&self, content: &str) -> Result<TranscriptionResponse> {
        // whisper-cli JSON format:
        // {
        //   "transcription": [
        //     { "timestamps": { "from": "00:00:00", "to": "00:00:03" }, "offsets": { "from": 0, "to": 3000 }, "text": "..." }
        //   ]
        // }
        // Or the newer format:
        // { "text": "...", "language": "en", "segments": [...] }

        #[derive(serde::Deserialize)]
        struct WhisperJson {
            #[serde(default)]
            text: Option<String>,
            #[serde(default)]
            language: Option<String>,
            #[serde(default)]
            segments: Option<Vec<WhisperSegment>>,
            #[serde(default)]
            transcription: Option<Vec<WhisperSegment>>,
        }

        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct WhisperSegment {
            #[serde(default)]
            text: Option<String>,
            #[serde(default)]
            timestamps: Option<WhisperTimestamps>,
            #[serde(default)]
            offsets: Option<WhisperOffsets>,
            // Newer format fields
            #[serde(default, rename = "start")]
            start_f64: Option<f64>,
            #[serde(default, rename = "end")]
            end_f64: Option<f64>,
            #[serde(default)]
            id: Option<u32>,
        }

        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct WhisperTimestamps {
            from: String,
            to: String,
        }

        #[derive(serde::Deserialize)]
        struct WhisperOffsets {
            from: u64,
            to: u64,
        }

        let parsed: WhisperJson = serde_json::from_str(content)
            .map_err(|e| AthenasError::Backend(format!("Failed to parse whisper JSON: {}", e)))?;

        // Extract text
        let text = parsed.text.unwrap_or_else(|| {
            // If no top-level text, concatenate from segments
            let segments = parsed.transcription.as_ref().or(parsed.segments.as_ref());
            segments
                .map(|segs| {
                    segs.iter()
                        .filter_map(|s| s.text.as_ref())
                        .map(|t| t.trim())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default()
        });

        // Extract segments
        let raw_segments = parsed.transcription.or(parsed.segments).unwrap_or_default();
        let segments: Vec<TranscriptionSegment> = raw_segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (start, end) = if let (Some(start), Some(end)) = (s.start_f64, s.end_f64) {
                    (start, end)
                } else if let Some(ref offsets) = s.offsets {
                    (offsets.from as f64 / 1000.0, offsets.to as f64 / 1000.0)
                } else {
                    (0.0, 0.0)
                };
                TranscriptionSegment {
                    id: s.id.unwrap_or(i as u32),
                    start,
                    end,
                    text: s.text.clone().unwrap_or_default().trim().to_string(),
                }
            })
            .collect();

        Ok(TranscriptionResponse {
            text: text.trim().to_string(),
            language: parsed.language,
            duration: None,
            segments,
        })
    }

    /// Parse SRT or VTT subtitle output.
    fn parse_subtitle_output(content: &str, format: &str) -> TranscriptionResponse {
        let segments = if format == "srt" {
            Self::parse_srt(content)
        } else {
            Self::parse_vtt(content)
        };

        let text = segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");

        TranscriptionResponse {
            text,
            language: None,
            duration: None,
            segments,
        }
    }

    /// Parse SRT format subtitles.
    fn parse_srt(content: &str) -> Vec<TranscriptionSegment> {
        let mut segments = Vec::new();
        let blocks: Vec<&str> = content.trim().split("\n\n").collect();

        for (i, block) in blocks.iter().enumerate() {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                continue;
            }
            // Line 0: index, Line 1: timestamps, Line 2+: text
            let time_line = lines.get(1).unwrap_or(&"");
            let text = lines[2..].join(" ");
            let (start, end) = Self::parse_srt_timestamp(time_line);
            segments.push(TranscriptionSegment {
                id: i as u32,
                start,
                end,
                text: text.trim().to_string(),
            });
        }
        segments
    }

    /// Parse VTT format subtitles.
    fn parse_vtt(content: &str) -> Vec<TranscriptionSegment> {
        let mut segments = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        // Skip WEBVTT header
        while i < lines.len() && (lines[i].is_empty() || lines[i].starts_with("WEBVTT")) {
            i += 1;
        }

        let mut seg_id = 0u32;
        while i < lines.len() {
            let line = lines[i];
            // Look for timestamp line: 00:00:00.000 --> 00:00:03.000
            if line.contains("-->") {
                let (start, end) = Self::parse_vtt_timestamp(line);
                i += 1;
                let mut text_parts = Vec::new();
                while i < lines.len() && !lines[i].is_empty() && !lines[i].contains("-->") {
                    text_parts.push(lines[i]);
                    i += 1;
                }
                segments.push(TranscriptionSegment {
                    id: seg_id,
                    start,
                    end,
                    text: text_parts.join(" ").trim().to_string(),
                });
                seg_id += 1;
            } else {
                i += 1;
            }
        }
        segments
    }

    /// Parse SRT timestamp: "00:00:01,234 --> 00:00:03,456"
    fn parse_srt_timestamp(line: &str) -> (f64, f64) {
        let parts: Vec<&str> = line.split("-->").collect();
        if parts.len() != 2 {
            return (0.0, 0.0);
        }
        let start = Self::time_str_to_secs(parts[0].trim(), ',');
        let end = Self::time_str_to_secs(parts[1].trim(), ',');
        (start, end)
    }

    /// Parse VTT timestamp: "00:00:01.234 --> 00:00:03.456"
    fn parse_vtt_timestamp(line: &str) -> (f64, f64) {
        let parts: Vec<&str> = line.split("-->").collect();
        if parts.len() != 2 {
            return (0.0, 0.0);
        }
        let start = Self::time_str_to_secs(parts[0].trim(), '.');
        let end = Self::time_str_to_secs(parts[1].trim(), '.');
        (start, end)
    }

    /// Convert "HH:MM:SS,mmm" or "HH:MM:SS.mmm" to seconds.
    fn time_str_to_secs(s: &str, _sep: char) -> f64 {
        let s = s.trim();
        // Handle both comma and dot as millisecond separator
        let s = s.replace(',', ".");
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() == 3 {
            let h: f64 = parts[0].parse().unwrap_or(0.0);
            let m: f64 = parts[1].parse().unwrap_or(0.0);
            let sec: f64 = parts[2].parse().unwrap_or(0.0);
            h * 3600.0 + m * 60.0 + sec
        } else if parts.len() == 2 {
            let m: f64 = parts[0].parse().unwrap_or(0.0);
            let sec: f64 = parts[1].parse().unwrap_or(0.0);
            m * 60.0 + sec
        } else {
            s.parse().unwrap_or(0.0)
        }
    }
}
