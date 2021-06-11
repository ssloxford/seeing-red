//
//  Defaults.swift
//  HeartbeatApp
//
//  Created by Henry Turner on 24/06/2019.
//  Copyright © 2019 SSL Oxford. All rights reserved.
//

import Foundation


struct defaultsKeys {
    // for holding data
    static let registrationDetails = "registrationDetails"
    static let UniqueUserID = "serverProvidedUniqueUserID"
    static let NumberOfReadingsCompeleted = "numberOfReadingsSuccesfullySentToServer"
    static let mturkFirstMeasurementValidationCode = "mturkValidationCode"
    static let pastTransactions = "pastTransactionsDictionary"
    static let PushNotificationsAccepted = "hasAcceptedPushNotifications"
    static let RetryURL = "urlToBeRetried"
    static let TimeToNextReading = "timeToNextReading"
    static let BlacklistedReason = "reasonForBlacklisting"
    static let CaptureSessionFromNotification = "captureSessionFromNotification"
    // status trackers
    static let GeneralStatus = "generalStatus"
}

enum GeneralStatus: Int {
    case uploadInProgress = 1
    case submissionAvailable = 2
    case noSubmissionAvailable = 3
    case blacklisted = 4
    case notRegistered = 5
    case firstMeasurementNeeded = 6
}

enum UploadStatus {
    case currentlyUploading
    case lastUploadAccepted
    case lastUploadRejected
}


struct apiEndpoints {
    static let registration = ""
    static let paymentHistory = ""
    static let timeToNext = ""
    static let upload = ""
    
}

struct testingMode {
    static let enabled = true
}

struct defaultParams {
    static let targetVideoFrameRate : Double = 240
    static let maximumAccelerationThreshold: Double = 1.3
}
