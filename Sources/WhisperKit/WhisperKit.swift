import Foundation
import whisper

public struct WhisperTranscription: Sendable {
    public let text: String
    public let duration: TimeInterval
    public let processingTime: TimeInterval
    public let segmentCount: Int
    public let samplePeak: Float
    public let sampleRMS: Float

    public init(
        text: String,
        duration: TimeInterval,
        processingTime: TimeInterval,
        segmentCount: Int,
        samplePeak: Float,
        sampleRMS: Float
    ) {
        self.text = text
        self.duration = duration
        self.processingTime = processingTime
        self.segmentCount = segmentCount
        self.samplePeak = samplePeak
        self.sampleRMS = sampleRMS
    }
}

public final class WhisperModel {
    private let context: OpaquePointer

    public init(modelPath: String, useGPU: Bool = true) throws {
        var params = whisper_context_default_params()
        params.use_gpu = useGPU
        guard let context = whisper_init_from_file_with_params(modelPath, params) else {
            throw WhisperError.loadFailed(modelPath)
        }
        self.context = context
    }

    deinit {
        whisper_free(context)
    }

    public func transcribe(samples: [Float], threadCount: Int) throws -> WhisperTranscription {
        let startedAt = Date()
        guard !samples.isEmpty else {
            return WhisperTranscription(
                text: "",
                duration: 0,
                processingTime: 0,
                segmentCount: 0,
                samplePeak: 0,
                sampleRMS: 0
            )
        }
        let stats = sampleStats(samples)

        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_special = false
        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false
        params.no_context = true
        params.single_segment = false
        params.no_timestamps = false
        params.language = nil
        params.detect_language = false
        params.n_threads = Int32(max(1, threadCount))

        let result = samples.withUnsafeBufferPointer { buffer in
            whisper_full(context, params, buffer.baseAddress, Int32(buffer.count))
        }
        guard result == 0 else {
            throw WhisperError.transcriptionFailed(result)
        }

        let segmentCount = whisper_full_n_segments(context)
        var text = ""
        for index in 0..<segmentCount {
            guard let segmentText = whisper_full_get_segment_text(context, index) else { continue }
            text += String(cString: segmentText)
        }

        return WhisperTranscription(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            duration: Double(samples.count) / 16_000,
            processingTime: Date().timeIntervalSince(startedAt),
            segmentCount: Int(segmentCount),
            samplePeak: stats.peak,
            sampleRMS: stats.rms
        )
    }

    private func sampleStats(_ samples: [Float]) -> (peak: Float, rms: Float) {
        var peak: Float = 0
        var sumSquares: Double = 0
        for sample in samples {
            let absolute = abs(sample)
            peak = max(peak, absolute)
            sumSquares += Double(sample * sample)
        }
        return (peak, Float(sqrt(sumSquares / Double(samples.count))))
    }
}

public enum WhisperError: LocalizedError {
    case loadFailed(String)
    case transcriptionFailed(Int32)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let modelPath):
            "Failed to load Whisper model: \(URL(fileURLWithPath: modelPath).lastPathComponent)."
        case .transcriptionFailed(let code):
            "Whisper transcription failed with code \(code)."
        }
    }
}
