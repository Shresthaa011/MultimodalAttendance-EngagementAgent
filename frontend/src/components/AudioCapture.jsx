import React, { useEffect, useRef, useState } from 'react';

const AudioCapture = ({ active, onMetrics }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const inputRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    if (active) {
      startCapture();
    } else {
      stopCapture();
    }
    return () => stopCapture();
  }, [active]);

  const startCapture = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false, // retain natural ambient sound for environment analytics
        },
      });
      streamRef.current = stream;
      setIsRecording(true);

      // Connect WS
      const ws = new WebSocket('ws://localhost:8000/ws/audio');
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log('[AudioCapture] WebSocket connected to ws://localhost:8000/ws/audio');

        // Initialize AudioContext
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioContext({ sampleRate: 16000 });
        audioContextRef.current = audioCtx;

        const source = audioCtx.createMediaStreamSource(stream);
        inputRef.current = source;

        // ScriptProcessor with 4096 samples (~256 ms per buffer at 16 kHz)
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;

          const inputData = e.inputBuffer.getChannelData(0);
          // Convert Float32Array to 16-bit PCM Int16Array
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          ws.send(pcm16.buffer);
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'audio_metrics') {
            setMetrics(data.metrics);
            if (onMetrics) {
              onMetrics(data);
            }
          }
        } catch (err) {
          console.error('[AudioCapture] Error parsing audio telemetry:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('[AudioCapture] WebSocket error:', err);
        setError('WebSocket error connecting to audio service');
      };

      ws.onclose = () => {
        console.log('[AudioCapture] WebSocket connection closed');
        setIsRecording(false);
      };
    } catch (err) {
      console.error('[AudioCapture] Error accessing microphone:', err);
      setError('Microphone permission denied or audio device unavailable');
      setIsRecording(false);
    }
  };

  const stopCapture = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (inputRef.current) {
      inputRef.current.disconnect();
      inputRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
    setIsRecording(false);
  };

  return (
    <div className="audio-capture-container p-4 bg-slate-900 border border-slate-700 rounded-xl text-white shadow-lg">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className={`w-3 h-3 rounded-full ${
              isRecording ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'
            }`}
          />
          <h3 className="font-semibold text-sm tracking-wide">
            {isRecording ? 'Ambient Audio Ingestion Active' : 'Audio Ingestion Idle'}
          </h3>
        </div>
        <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
          16 kHz PCM
        </span>
      </div>

      {error && (
        <div className="p-2 mb-3 bg-red-900/60 border border-red-700 rounded text-xs text-red-200">
          {error}
        </div>
      )}

      {metrics && (
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div className="bg-slate-800 p-2 rounded">
            <span className="text-slate-400 block text-[10px]">Noise Level</span>
            <span className="text-sm font-bold text-amber-400">
              {metrics.noise_level_db} dBFS
            </span>
          </div>
          <div className="bg-slate-800 p-2 rounded">
            <span className="text-slate-400 block text-[10px]">Speech Activity</span>
            <span
              className={`text-sm font-bold ${
                metrics.speech_detected ? 'text-emerald-400' : 'text-slate-400'
              }`}
            >
              {metrics.speech_detected ? 'Active Speech' : 'Silence / Listening'}
            </span>
          </div>
          <div className="bg-slate-800 p-2 rounded">
            <span className="text-slate-400 block text-[10px]">Speech Ratio</span>
            <span className="text-sm font-bold text-sky-400">
              {(metrics.speech_ratio * 100).toFixed(1)}%
            </span>
          </div>
          <div className="bg-slate-800 p-2 rounded">
            <span className="text-slate-400 block text-[10px]">Acoustic Engagement</span>
            <span className="text-sm font-bold text-violet-400">
              {(metrics.acoustic_engagement_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioCapture;

