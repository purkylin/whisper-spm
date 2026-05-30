// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "Whisper",
    platforms: [
        .iOS("16.4"),
        .macOS("13.3"),
        .tvOS("16.4"),
        .visionOS("1.0"),
    ],
    products: [
        .library(
            name: "Whisper",
            targets: ["Whisper"]
        ),
    ],
    targets: [
        .target(
            name: "Whisper",
            dependencies: ["WhisperFramework"]
        ),
        .binaryTarget(
            name: "WhisperFramework",
            url: "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.5/whisper-v1.8.5-xcframework.zip",
            checksum: "6e7ffd33abc447758d05546c2c151c0bd58cc1ebd495e0bf17583178845b34bd"
        ),
    ],
    swiftLanguageModes: [.v6]
)
