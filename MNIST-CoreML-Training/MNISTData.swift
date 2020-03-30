//
//  MNISTData.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 3/29/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation

public struct MNISTData {
    public let numTrainRecords: Int
    public let numTestRecords: Int
    
    public let xTrain: [[Float]]
    public let yTrain: [[Int]]

    public let xTest: [[Float]]
    public let yTest: [[Int]]
    
    static func processFile(path: URL) -> (numRecords: Int, image: [[Float]], label: [[Int]]) {
        func oneHotEncode(_ n: Int) -> [Int] {
            var encode = Array(repeating: 0, count: 10)
            encode[n] = 1
            return encode
        }
        
        // Load Data
        let data = try! String(contentsOf: path, encoding: String.Encoding.utf8)

        // Convert Space Separated CSV with no Header
        var image = [[Float]]()
        var label = [[Int]]()
        
        data.split(separator: "\r\n")
            .map{ String($0).split(separator: ",").map{ Int(String($0))! } }
            .forEach{ (intList: [Int]) in
                image.append(Array(intList[1...].map{ Float($0) / Float(255.0) }))
                label.append(oneHotEncode(intList[0]))
            }
        
        return (image.count, image, label)
    }

    public init() {
        let trainFilePath = Bundle.main.url(forResource: "mnist_train", withExtension: "csv")!
        let testFilePath = Bundle.main.url(forResource: "mnist_test", withExtension: "csv")!

        (self.numTrainRecords, self.xTrain, self.yTrain) = MNISTData.processFile(path: trainFilePath)
        (self.numTestRecords, self.xTest, self.yTest) = MNISTData.processFile(path: testFilePath)
    }
}
