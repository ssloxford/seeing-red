//
//  ViewController.swift
//  HeartbeatApp
//
//  Created by Henry Turner on 19/06/2019.
//  Copyright © 2019 SSL Oxford. All rights reserved.
//


import UIKit

class ViewController: UIViewController {
    
    
    @IBOutlet weak var NextActionButton: UIButton!
        
    override func viewDidLoad() {
        super.viewDidLoad()
                
    }
    
    override func viewDidAppear(_ animated: Bool) {
        
    }
    
    @IBAction func beginMeasurement(_ sender: Any) {
        
        self.loadMeasurementView()
    }
        
    fileprivate func loadMeasurementView() {
        
        let storyBoard: UIStoryboard = UIStoryboard(name: "Main", bundle: nil)
        let nextVC : UIViewController = storyBoard.instantiateViewController(withIdentifier: "capture") as! CaptureViewController
        self.navigationController?.pushViewController(nextVC, animated: true)
    }
    
}
