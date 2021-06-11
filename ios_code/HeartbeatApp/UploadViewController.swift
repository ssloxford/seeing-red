//
//  UploadViewController.swift
//  HeartbeatApp
//
//  Created by Henry Turner on 16/07/2019.
//  Copyright © 2019 SSL Oxford. All rights reserved.
//

import Foundation
import UIKit
import AVFoundation
import CoreMedia
import Alamofire
import SwiftUI

class UploadViewController : UIViewController {
    
    @IBOutlet weak var ActionLabel: UILabel!
    @IBOutlet weak var UploadIndicator: UIActivityIndicatorView!
    @IBOutlet weak var ActionIcon: UIImageView!
    
    @IBOutlet weak var HomeButton: UIButton!
    
    @IBOutlet weak var TestingLabel: UILabel!
    @IBOutlet weak var TestingTextbox: UITextField!
    @IBOutlet weak var TestingButton: UIButton!
    
    
    var videoPath: URL?
    var scaledPath: URL?
    //    var exportSession: AVAssetExportSession?
    var videoAsset : AVAsset?
    var accelData:String? = nil
    private var exportSession: VIExportSession!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.UploadIndicator.hidesWhenStopped = true

        ActionLabel.text = ""
        
        if testingMode.enabled {
            // Display the extra field to be filled in as optional
            TestingLabel.isHidden = false
            TestingTextbox.isHidden = false
            TestingButton.isHidden = false
            let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: "dismissKeyboard")
            view.addGestureRecognizer(tap)
            NotificationCenter.default.addObserver(self, selector: #selector(keyboardWillShow), name: UIResponder.keyboardWillShowNotification, object: nil)
            NotificationCenter.default.addObserver(self, selector: #selector(keyboardWillHide), name: UIResponder.keyboardWillHideNotification, object: nil)
            
        }
        else {
            TestingLabel.isHidden = true
            TestingTextbox.isHidden = true
            TestingButton.isHidden = true
        }
        
        
    }
    
    @objc func keyboardWillShow(notification: NSNotification) {
        if let keyboardSize = (notification.userInfo?[UIResponder.keyboardFrameBeginUserInfoKey] as? NSValue)?.cgRectValue {
            if self.view.frame.origin.y == 0 {
                self.view.frame.origin.y -= keyboardSize.height
            }
        }
    }
    
    @objc func keyboardWillHide(notification: NSNotification) {
        if self.view.frame.origin.y != 0 {
            self.view.frame.origin.y = 0
        }
    }
    
    
    override func viewDidAppear(_ animated: Bool) {
        print("About to check video")
        exportSession = nil
        if !testingMode.enabled {
            checkVideo()
        }
        else {
            print("testing mode enabled, will not auto upload")
        }
    }
    
    
    @IBAction func StartUpload(_ sender: Any) {
        checkVideo()
    }
    
    fileprivate func checkVideo() {
        // functions to check the video go here
        // we should probably check it has some length and report that there has been a bug if so
        ActionLabel.text = "Compressing video"
        ActionIcon.image = UIImage(systemName: "gear")
        
        let asset = AVAsset(url: videoPath!)
        
        do {
            let resources = try videoPath!.resourceValues(forKeys:[.fileSizeKey])
            let fileSize = resources.fileSize!
            print ("Filesize: \(fileSize)")
        } catch {
            print("Error: \(error)")
        }
        
        let duration = asset.duration
        let durationTime = CMTimeGetSeconds(duration)
        
        print("Duration: \(durationTime)")
        print("Resolution: \(asset.tracks(withMediaType: AVMediaType.video).first!.naturalSize)")
        print("FPS: \(asset.tracks(withMediaType: AVMediaType.video).first!.nominalFrameRate)")
        
        if durationTime <= 1 {
            // video is not the required length, display failure image and send back to the capture page
            let alertController = UIAlertController(title: "Alert", message: "There is an issue with the video, please try again later. If the problem persists contact us with the email on the homepage", preferredStyle: .alert)
            alertController.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: { (alertAction) -> Void in  }))
            self.present(alertController, animated: true, completion: nil)
        }
        
        // scale the video here
        scaleVideo(inputVideo: videoPath!)
        
    }
    
    func configureExportConfiguration(videoTrack: AVAssetTrack) {
        exportSession.videoConfiguration.videoOutputSetting = {
            let frameRate = defaultParams.targetVideoFrameRate
            let bitrate = min(5000000, videoTrack.estimatedDataRate) //2Mbps
            let trackDimensions = CGSize(width: 360, height: 240)
            let compressionSettings: [String: Any] = [
                AVVideoAverageNonDroppableFrameRateKey: frameRate,
                AVVideoAverageBitRateKey: bitrate,
                AVVideoMaxKeyFrameIntervalKey: 240,
                //                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
                AVVideoProfileLevelKey: AVVideoProfileLevelH264MainAutoLevel
            ]
            var videoSettings: [String : Any] = [
                AVVideoWidthKey: trackDimensions.width,
                AVVideoHeightKey: trackDimensions.height,
                AVVideoCompressionPropertiesKey: compressionSettings
            ]
            if #available(iOS 11.0, *) {
                videoSettings[AVVideoCodecKey] =  AVVideoCodecType.h264
            } else {
                videoSettings[AVVideoCodecKey] =  AVVideoCodecH264
            }
            return videoSettings
        }()
    }
    
    fileprivate func scaleVideo(inputVideo : URL) {
        
        self.videoAsset = AVAsset(url: inputVideo)
        print("Export asset duration")
        print(self.videoAsset!.duration)
        print(self.videoAsset?.isExportable)
        
        exportSession = VIExportSession.init(asset: self.videoAsset!)
        if let track = self.videoAsset!.tracks(withMediaType: .video).first {
            configureExportConfiguration(videoTrack: track)
        }
        
        scaledPath = exportSession.exportConfiguration.outputURL
        
        exportSession.completionHandler = { [weak self] (error) in
            guard let strongSelf = self else { return }
            DispatchQueue.main.async {
                if let error = error {
                    print(error.localizedDescription)
                } else {
                    print("Exporting to new format now complete, can do what you like")
                }
            }
        }
        exportSession.export()
    }
    
        
    @IBAction func ReturnHome(_ sender: Any) {
        self.navigationController?.popToRootViewController(animated: true)
    }
    
    @objc func dismissKeyboard() {
        //Causes the view (or one of its embedded text fields) to resign the first responder status.
        view.endEditing(true)
    }
}
