//
//  VIExportSession.swift
//  VIExportSession
//
//  Created by Vito on 30/01/2018.
//  Copyright © 2018 Vito. All rights reserved.
//

import AVFoundation

public class VIExportSession {
    
    public private(set) var asset: AVAsset
    public var exportConfiguration = ExportConfiguration()
    public var videoConfiguration = VideoConfiguration()
    public var audioConfiguration = AudioConfiguration()
    
    fileprivate var reader: AVAssetReader!
    fileprivate var videoOutput: AVAssetReaderVideoCompositionOutput?
    fileprivate var audioOutput: AVAssetReaderAudioMixOutput?
    fileprivate var writer: AVAssetWriter!
    fileprivate var videoInput: AVAssetWriterInput?
    fileprivate var audioInput: AVAssetWriterInput?
    fileprivate var inputQueue = DispatchQueue(label: "VideoEncoderQueue")
    
    // MARK: - Exporting properties
    public var progress: Float = 0 {
        didSet {
            progressHandler?(progress)
        }
    }
    public var videoProgress: Float = 0 {
        didSet {
            if audioInput != nil {
                progress = 0.95 * videoProgress + 0.05 * audioProgress
            } else {
                progress = videoProgress
            }
        }
    }
    public var audioProgress: Float = 0 {
        didSet {
            if videoInput != nil {
                progress = 0.95 * videoProgress + 0.05 * audioProgress
            } else {
                progress = audioProgress
            }
        }
    }
    
    public var progressHandler: ((Float) -> Void)?
    public var completionHandler: ((Error?) -> Void)?
    
    fileprivate var videoCompleted = false
    fileprivate var audioCompleted = false
    
    public init(asset: AVAsset) {
        self.asset = asset
    }
    
    // MARK: - Main
    
    public func cancelExport() {
        if let writer = writer, let reader = reader {
            inputQueue.async {
                writer.cancelWriting()
                reader.cancelReading()
            }
        }
    }
    
    public func export() {
        cancelExport()
        reset()
        do {
            reader = try AVAssetReader(asset: asset)
            writer = try AVAssetWriter(url: exportConfiguration.outputURL, fileType: exportConfiguration.fileType)
            
            writer.shouldOptimizeForNetworkUse = exportConfiguration.shouldOptimizeForNetworkUse
            writer.metadata = exportConfiguration.metadata
            
            // Video output
            let videoTracks = asset.tracks(withMediaType: .video)
            if videoTracks.count > 0 {
                if videoConfiguration.videoOutputSetting == nil {
                    videoConfiguration.videoOutputSetting = buildDefaultVideoOutputSetting(videoTrack: videoTracks.first!)
                }
                
                let videoOutput = AVAssetReaderVideoCompositionOutput(videoTracks: videoTracks, videoSettings: videoConfiguration.videoInputSetting)
                videoOutput.alwaysCopiesSampleData = false
                videoOutput.videoComposition = videoConfiguration.videoComposition
                if videoOutput.videoComposition == nil {
                    videoOutput.videoComposition = buildDefaultVideoComposition(with: asset)
                }
                
                guard reader.canAdd(videoOutput) else {
                    throw NSError(domain: "com.exportsession", code: 0, userInfo: [NSLocalizedDescriptionKey: NSLocalizedString("Can't add video output", comment: "")])
                }
                reader.add(videoOutput)
                self.videoOutput = videoOutput
                
                // Video input
                let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoConfiguration.videoOutputSetting)
                videoInput.expectsMediaDataInRealTime = false
                guard writer.canAdd(videoInput) else {
                    throw NSError(domain: "com.exportsession", code: 0, userInfo: [NSLocalizedDescriptionKey: NSLocalizedString("Can't add video input", comment: "")])
                }
                writer.add(videoInput)
                self.videoInput = videoInput
            }
            
            // Audio output
            let audioTracks = asset.tracks(withMediaType: .audio)
            if audioTracks.count > 0 {
                if audioConfiguration.audioOutputSetting == nil {
                    audioConfiguration.audioOutputSetting = buildDefaultAudioOutputSetting()
                }
                
                let audioOutput = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: audioConfiguration.audioInputSetting)
                audioOutput.alwaysCopiesSampleData = false
                audioOutput.audioMix = audioConfiguration.audioMix
                if let audioTimePitchAlgorithm = audioConfiguration.audioTimePitchAlgorithm {
                    audioOutput.audioTimePitchAlgorithm = audioTimePitchAlgorithm
                }
                if reader.canAdd(audioOutput) {
                    reader.add(audioOutput)
                    self.audioOutput = audioOutput
                }
                
                if self.audioOutput != nil {
                    // Audio input
                    let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioConfiguration.audioOutputSetting)
                    audioInput.expectsMediaDataInRealTime = false
                    if writer.canAdd(audioInput) {
                        writer.add(audioInput)
                        self.audioInput = audioInput
                    }
                }
            }
            
            writer.startWriting()
            reader.startReading()
            writer.startSession(atSourceTime: CMTime.zero)
            
            encodeVideoData()
            encodeAudioData()
        } catch {
            self.completionHandler?(error)
        }
    }
    
    fileprivate func encodeVideoData() {
        if let videoInput = videoInput {
            videoInput.requestMediaDataWhenReady(on: inputQueue, using: { [weak self] in
                guard let strongSelf = self else { return }
                guard let videoOutput = strongSelf.videoOutput, let videoInput = strongSelf.videoInput else { return }
                strongSelf.encodeReadySamplesFrom(output: videoOutput, to: videoInput, completion: {
                    strongSelf.videoCompleted = true
                    strongSelf.tryFinish()
                })
            })
        } else {
            videoCompleted = true
            tryFinish()
        }
    }
    
    fileprivate func encodeAudioData() {
        if let audioInput = audioInput {
            audioInput.requestMediaDataWhenReady(on: inputQueue, using: { [weak self] in
                guard let strongSelf = self else { return }
                guard let audioOutput = strongSelf.audioOutput, let audioInput = strongSelf.audioInput else { return }
                strongSelf.encodeReadySamplesFrom(output: audioOutput, to: audioInput, completion: {
                    strongSelf.audioCompleted = true
                    strongSelf.tryFinish()
                })
            })
        } else {
            audioCompleted = true
            tryFinish()
        }
    }
    
    private var lastVideoSamplePresentationTime = CMTime.zero
    private var lastAudioSamplePresentationTime = CMTime.zero
    fileprivate func encodeReadySamplesFrom(output: AVAssetReaderOutput, to input: AVAssetWriterInput, completion: @escaping () -> Void) {
        while input.isReadyForMoreMediaData {
            let complete = autoreleasepool(invoking: { [weak self] () -> Bool in
                guard let strongSelf = self else { return true }
                if let sampleBuffer = output.copyNextSampleBuffer() {
                    guard strongSelf.reader.status == .reading && strongSelf.writer.status == .writing else {
                        return true
                    }
                    
                    guard input.append(sampleBuffer) else {
                        return true
                    }
                    
                    if let videoOutput = strongSelf.videoOutput, videoOutput == output {
                        lastVideoSamplePresentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                        if strongSelf.asset.duration.seconds > 0 {
                            strongSelf.videoProgress = Float(lastVideoSamplePresentationTime.seconds / strongSelf.asset.duration.seconds)
                        } else {
                            strongSelf.videoProgress = 1
                        }
                    } else if let audioOutput = strongSelf.audioOutput, audioOutput == output {
                        lastAudioSamplePresentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                        if strongSelf.asset.duration.seconds > 0 {
                            strongSelf.audioProgress = Float(lastAudioSamplePresentationTime.seconds / strongSelf.asset.duration.seconds)
                        } else {
                            strongSelf.audioProgress = 1
                        }
                    }
                } else {
                    input.markAsFinished()
                    return true
                }
                return false
            })
            if complete {
                completion()
                break
            }
        }
    }
    
    fileprivate func tryFinish() {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        if audioCompleted && videoCompleted {
            if reader.status == .cancelled || writer.status == .cancelled {
                finish()
            } else if writer.status == .failed {
                finish()
            } else if reader.status == .failed {
                writer.cancelWriting()
                finish()
            } else {
                writer.finishWriting { [weak self] in
                    guard let strongSelf = self else { return }
                    strongSelf.finish()
                }
            }
        }
    }
    
    fileprivate func finish() {
        if writer.status == .failed || reader.status == .failed {
            try? FileManager.default.removeItem(at: exportConfiguration.outputURL)
        }
        let error = writer.error ?? reader.error
        completionHandler?(error)
        
        reset()
    }
    
    fileprivate func reset() {
        videoCompleted = false
        videoCompleted = false
        
        videoProgress = 0
        audioProgress = 0
        progress = 0
        
        reader = nil
        videoOutput = nil
        audioInput = nil
        writer = nil
        videoInput = nil
        audioInput = nil
    }
    
}

extension VIExportSession {
    fileprivate func buildDefaultVideoComposition(with asset: AVAsset) -> AVVideoComposition {
        let videoComposition = AVMutableVideoComposition()
        
        if let videoTrack = asset.tracks(withMediaType: .video).first {
            var trackFrameRate: Float = 30
            if let videoCompressionProperties = videoConfiguration.videoOutputSetting?[AVVideoCompressionPropertiesKey] as? [String: Any],
                let frameRate = videoCompressionProperties[AVVideoAverageNonDroppableFrameRateKey] as? NSNumber {
                trackFrameRate = frameRate.floatValue
            } else {
                trackFrameRate = videoTrack.nominalFrameRate
            }
            videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(trackFrameRate))
            
            var naturalSize = videoTrack.naturalSize
            var transform = videoTrack.preferredTransform
            let angle = atan2(transform.b, transform.a)
            let videoAngleInDegree = angle * 180 / CGFloat.pi
            if videoAngleInDegree == 90 || videoAngleInDegree == -90 {
                let width = naturalSize.width
                naturalSize.width = naturalSize.height
                naturalSize.height = width
            }
            
            videoComposition.renderSize = naturalSize
            
            var targetSize = naturalSize
            if let width = videoConfiguration.videoOutputSetting?[AVVideoWidthKey] as? NSNumber {
                targetSize.width = CGFloat(width.floatValue)
            }
            if let height = videoConfiguration.videoOutputSetting?[AVVideoHeightKey] as? NSNumber {
                targetSize.height = CGFloat(height.floatValue)
            }
            // Center
            if naturalSize.width > 0 && naturalSize.height > 0 {
                let xratio = targetSize.width / naturalSize.width
                let yratio = targetSize.height / naturalSize.height
                let ratio = min(xratio, yratio)
                let postWidth = naturalSize.width * ratio
                let postHeight = naturalSize.height * ratio
                let transx = (targetSize.width - postWidth) * 0.5
                let transy = (targetSize.height - postHeight) * 0.5
                var matrix = CGAffineTransform(translationX: transx / xratio, y: transy / yratio)
                matrix = matrix.scaledBy(x: ratio / xratio, y: ratio / yratio)
                transform = transform.concatenating(matrix)
            }
            
            let passThroughInstruction = AVMutableVideoCompositionInstruction()
            passThroughInstruction.timeRange = CMTimeRangeMake(start:CMTime.zero, duration:asset.duration)
            let passThroughLayer = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
            passThroughLayer.setTransform(transform, at: CMTime.zero)
            passThroughInstruction.layerInstructions = [passThroughLayer]
            videoComposition.instructions = [passThroughInstruction]
        }
        
        return videoComposition
    }
    
    fileprivate func buildDefaultVideoOutputSetting(videoTrack: AVAssetTrack) -> [String: Any] {
        let trackDimensions = { () -> CGSize in
            var trackDimensions = videoTrack.naturalSize
            let videoAngleInDegree = atan2(videoTrack.preferredTransform.b, videoTrack.preferredTransform.a) * 180.0 / CGFloat(Double.pi)
            if abs(videoAngleInDegree) == 90 {
                let width = trackDimensions.width
                trackDimensions.width = trackDimensions.height
                trackDimensions.height = width
            }
            
            return trackDimensions
        }()
        
        var videoSettings: [String : Any] = [
            AVVideoWidthKey: trackDimensions.width,
            AVVideoHeightKey: trackDimensions.height,
        ]
        if #available(iOS 11.0, *) {
            videoSettings[AVVideoCodecKey] = AVVideoCodecType.h264
        } else {
            videoSettings[AVVideoCodecKey] = AVVideoCodecH264
        }
        return videoSettings
    }
    
    fileprivate func buildDefaultAudioOutputSetting() -> [String: Any] {
        var stereoChannelLayout = AudioChannelLayout()
        memset(&stereoChannelLayout, 0, MemoryLayout<AudioChannelLayout>.size)
        stereoChannelLayout.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo
        
        let channelLayoutAsData = Data(bytes: &stereoChannelLayout, count: MemoryLayout<AudioChannelLayout>.size)
        let compressionAudioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVEncoderBitRateKey: 128000,
            AVSampleRateKey: 44100,
            AVChannelLayoutKey: channelLayoutAsData,
            AVNumberOfChannelsKey: 2
        ]
        return compressionAudioSettings
    }
}

