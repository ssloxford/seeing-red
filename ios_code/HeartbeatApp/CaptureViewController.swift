//
//  CaptureViewController.swift
//  HeartbeatApp
//
//  Created by Henry Turner on 24/06/2019.
//  Copyright © 2019 SSL Oxford. All rights reserved.
//

import Foundation
import UIKit
//import CameraManager
import AVFoundation
import Alamofire
import CoreMotion

class CaptureViewController : UIViewController {
    
    let cameraManager = CameraManager()
    var timer:Timer?
    var accel_timer:Timer?
    let videoLength = 30
    var timeLeft = 0
    var cameraDisplayed = false
    
    var recordingBegun = false
    var recordingMustStop = false
    
    let motion = CMMotionManager()
    
    var accelerometer_readings:String? = ""

    
    // MARK: - @IBOutlets
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var cameraButton: UIButton!
    @IBOutlet weak var countdownLabel: UILabel!
    @IBOutlet weak var instructionImage: UIImageView!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        configureCorrectImageForInstructions()
        
        completeCameraSetup()
        startAccelerometers()
    }
    
    override func viewDidAppear(_ animated: Bool) {

    }
    
    fileprivate func configureCorrectImageForInstructions() {
        // TODO determine which iPhone model we are using
        // determine the camera layout of that iPhone model
        // display the corresponding image
        
        let iphone_camera_orientation = UIDevice.cameraLayout
        if iphone_camera_orientation == "sideways" {
            instructionImage.image = UIImage(named: "instructions-sideways")
        }
        else if iphone_camera_orientation == "vertical" {
            // layout must be vertical here
            instructionImage.image = UIImage(named: "instructions-vertical")
        }
        else if iphone_camera_orientation == "triangle" {
            instructionImage.image = UIImage(named: "instructions-triangle")
        }
        else {
            NSLog(iphone_camera_orientation)
            NSLog("what has happened here...")
        }
        
    }
    
    func startAccelerometers() {
        let frequency = defaultParams.targetVideoFrameRate
        // Make sure the accelerometer hardware is available.
        if self.motion.isAccelerometerAvailable {
            self.motion.accelerometerUpdateInterval = 1.0 / frequency  // 60 Hz
            self.motion.startAccelerometerUpdates()
            
            // Configure a timer to fetch the data.
            self.accel_timer = Timer(fire: Date(), interval: (1.0/frequency),
                               repeats: true, block: { (timer) in
                                // Get the accelerometer data.
                                if let data = self.motion.accelerometerData {
                                    let x = data.acceleration.x
                                    let y = data.acceleration.y
                                    let z = data.acceleration.z
                                    
                                    if self.recordingBegun {
                                        let total_accel = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
                                        if total_accel > defaultParams.maximumAccelerationThreshold {
                                            self.recordingMustStop = true
                                        }
                                        if self.accelerometer_readings != nil {
                                            let timestamp = NSDate().timeIntervalSince1970
                                            self.accelerometer_readings! += String(format:"%@,%@,%@,%@\n",
                                                                                       String(timestamp), String(x), String(y), String(z) )
                                        }
                                    }
                                }
            })
            
            // Add the timer to the current run loop.
            RunLoop.current.add(self.accel_timer!, forMode: RunLoop.Mode.default)
        }
    }
    
    func stopAccelerometers() {
        self.accel_timer?.invalidate()
        self.motion.stopAccelerometerUpdates()
    }
    
    deinit {
        stopAccelerometers()
    }
    
    
    fileprivate func completeCameraSetup() {
        // Do any additional setup after loading the view.
        timeLeft = videoLength
        cameraManager.shouldRespondToOrientationChanges = false
        cameraManager.cameraOutputMode = .videoOnly
        cameraManager.cameraDevice = .back
        cameraManager.cameraOutputQuality = .high
        cameraManager.flashMode = .on
        cameraManager.focusMode = .locked
        cameraManager.exposureMode = .custom
        cameraManager._changeExposureDuration(value: 0.9)
//        cameraManager.exposureValue = 0.5
        // TODO: change the iso somehow? <- need to extend the camera manager class for this, but should be
        
        cameraManager.videoStabilisationMode = .auto
        cameraManager.shouldUseLocationServices = false
        cameraManager.videoAlbumName = "HeartbeatVideos"
//        cameraManager.cameraOutputQuality = .low
        cameraManager.shouldEnableTapToFocus = false
        cameraManager.shouldEnableExposure = false
        cameraManager.shouldEnablePinchToZoom = false
        cameraManager.shouldFlipFrontCameraImage = false
        cameraManager.showAccessPermissionPopupAutomatically = true
        cameraManager.writeFilesToPhoneLibrary = false
        navigationController?.navigationBar.isHidden = true
        
        let currentCameraState = cameraManager.currentCameraStatus()
        
        if currentCameraState == .ready {
            print("showing camera preiview")
            addCameraToView()
        }
        else {
            cameraManager.askUserForCameraPermission({permissionGranted in
            
                if permissionGranted {
                    self.addCameraToView()
                }
                else{
                    print("User rejected, they will not be able to continue")
                }})
            
        }
    }
        
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        navigationController?.navigationBar.isHidden = true
        cameraManager.resumeCaptureSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraManager.stopCaptureSession()
    }
    
    // MARK: - ViewController
    fileprivate func addCameraToView()
    {
        if cameraDisplayed {
            return
        }
        cameraDisplayed = true
        cameraManager.addPreviewLayerToView(cameraView, newCameraOutputMode: CameraOutputMode.videoOnly)
        cameraManager.showErrorBlock = { [weak self] (erTitle: String, erMessage: String) -> Void in
            
            let alertController = UIAlertController(title: erTitle, message: erMessage, preferredStyle: .alert)
            alertController.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: { (alertAction) -> Void in  }))
            
            self?.present(alertController, animated: true, completion: nil)
        }
    }
    
    // MARK: - @IBActions
    
    @IBAction func backButton(_ sender: UIButton) {
        
        //TODO is this the appropriate way to disable the back button now the upload is separate?
        if timeLeft == videoLength {
            self.navigationController?.popToRootViewController(animated: true)
        }
        
    }
    
    @IBAction func recordButtonTapped(_ sender: UIButton) {
        
        if !cameraDisplayed {
            addCameraToView()
        }
        
        if cameraButton.isSelected {
            return
        }
        cameraButton.isSelected = !cameraButton.isSelected
        cameraButton.setTitle("Beginning recording", for: UIControl.State.selected)
        
        self.recordingBegun = true
        self.recordingMustStop = false
        self.accelerometer_readings = ""
        
        if sender.isSelected {
            timer = nil
            timer = Timer.scheduledTimer(timeInterval: 1.0, target: self, selector: #selector(onTimerFires), userInfo: nil, repeats: true)
            cameraManager.flashMode = .on
            cameraManager.startRecordingVideo()
        }
    }
    
    @objc func onTimerFires()
    {
        timeLeft -= 1
//        cameraButton.setTitle("\(timeLeft) seconds left", for: UIControl.State.selected)
        countdownLabel.text = String(timeLeft)
        
        if self.recordingMustStop {
            print("recording must stop")
            timer!.invalidate()
            timer = nil
            cameraManager.flashMode = .off
            timeLeft = videoLength
            countdownLabel.text = String(timeLeft)
            cameraButton.isSelected = false
            self.recordingMustStop = false
            self.recordingBegun = false
            cameraManager.stopVideoRecording({ (videoURL, error) -> Void in
                
                do {
                    try FileManager.default.removeItem(at: videoURL!)
                } catch let error as NSError {
                    print("Error: \(error.domain)")
                }
                
                let alertController = UIAlertController(title: "Please keep still!", message: "Too much movement was detected, please try again.", preferredStyle: .alert)
                alertController.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: { (alertAction) -> Void in  }))
                
                self.present(alertController, animated: true, completion: nil)
                
                
            })

        }
        
        if timeLeft <= 0 {
            timer!.invalidate()
            timer = nil
            timeLeft = videoLength
            cameraButton.isSelected = false
            cameraManager.stopVideoRecording({ (videoURL, error) -> Void in
                    self.cameraButton.setTitle("Beginning upload", for: UIControl.State.selected)
//                    print(videoURL)
                    let documentsUrl =  FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
                    let destinationPath = documentsUrl!.appendingPathComponent("video.mp4")
                

                    do {
                        try FileManager.default.removeItem(at: destinationPath)
                    } catch let error as NSError {
                        print("Error: \(error.domain)")
                    }
                
                    do {
                        try FileManager.default.moveItem(at: videoURL!, to: destinationPath)
                    } catch let error as NSError {
                        print("Error: \(error.domain)")
                    }
                
                        //todo upload the video here
                    let storyBoard: UIStoryboard = UIStoryboard(name: "Main", bundle: nil)
                    let uploadVC = storyBoard.instantiateViewController(withIdentifier: "upload") as! UploadViewController
                    uploadVC.videoPath = destinationPath
                    if self.accelerometer_readings != nil {
                        uploadVC.accelData = self.accelerometer_readings!
                    }
                    self.navigationController?.pushViewController(uploadVC, animated: true)
                
                
            })
            cameraManager.flashMode = .off
        }
        
    }
}
