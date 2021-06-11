//
//  ExportConfiguration.swift
//  VIExportSession
//
//  Created by Vito on 06/02/2018.
//  Copyright © 2018 Vito. All rights reserved.
//

import AVFoundation

public class ExportConfiguration {
    public var outputURL = URL.temporaryExportURL()
    public var fileType: AVFileType = .mp4
    public var shouldOptimizeForNetworkUse = false
    public var metadata: [AVMetadataItem] = []
}

public class VideoConfiguration {
    // Video settings see AVVideoSettings.h
    public var videoInputSetting: [String: Any]?
    public var videoOutputSetting: [String: Any]?
    public var videoComposition: AVVideoComposition?
}

public class AudioConfiguration {
    // Audio settings see AVAudioSettings.h
    public var audioInputSetting: [String: Any]?
    public var audioOutputSetting: [String: Any]?
    public var audioMix: AVAudioMix?
    public var audioTimePitchAlgorithm: AVAudioTimePitchAlgorithm?
}

// MARK: - Helper

fileprivate extension URL {
    static func temporaryExportURL() -> URL {
        let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).last!
        let filename = ProcessInfo.processInfo.globallyUniqueString + ".mp4"
        return documentDirectory.appendingPathComponent(filename)
    }
}
