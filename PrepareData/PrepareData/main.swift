//
//  main.swift
//  PrepareData
//
//  Created by Jacopo Mangiavacchi on 3/30/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

func oneHotEncode(_ n: Int) -> [Int] {
    var encode = Array(repeating: 0, count: 10)
    encode[n] = 1
    return encode
}

func processFile(path: URL) -> (image: [[Float]], label: [[Int]]) {
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
    
    return (image, label)
}

func processFileOptimized(path: URL) -> (image: [[Float]], label: [[Int]]) {
    var image = [[Float]]()
    var label = [[Int]]()

    errno = 0
    if freopen(path.path, "r", stdin) == nil {
        print("error opening file")
    }
    while let line = readLine()?.split(separator: ",") {
        label.append(oneHotEncode(Int(String(line[0]))!))
        image.append(Array(line[1...].map{ Float(String($0))! / Float(255.0) }))
    }

    return (image, label)
}

func prepareTrainingBatch(path: URL) -> [[String: [NSNumber]]] { // [MLFeatureProvider]
//    var featureProviders = [MLFeatureProvider]()
    var dictionaryArray = [[String: [NSNumber]]]()

    errno = 0
    if freopen(path.path, "r", stdin) == nil {
        print("error opening file")
    }
    while let line = readLine()?.split(separator: ",") {
//        let imageMultiArr = try! MLMultiArray(shape: [28, 28], dataType: .float32)
//        let outputMultiArr = try! MLMultiArray(shape: [10], dataType: .int32)
        var imageMultiArr = Array(repeating: NSNumber(value: 0.0), count: 28*28)
        var outputMultiArr = Array(repeating: NSNumber(value: 0), count: 10)

        // TODO imageMultiArr image.append(Array(line[1...].map{ Float(String($0))! / Float(255.0) }))
        for r in 0..<28 {
            for c in 0..<28 {
                let i = (r*28)+c
                imageMultiArr[i] = NSNumber(value: Float(String(line[i + 1]))! / Float(255.0))
            }
        }

        let oneHot = oneHotEncode(Int(String(line[0]))!)
        for i in 0..<10 {
            outputMultiArr[i] = NSNumber(value: oneHot[i])
        }
        
//        let imageValue = MLFeatureValue(multiArray: imageMultiArr)
//        let outputValue = MLFeatureValue(multiArray: outputMultiArr)

        let dataPointFeatures: [String: [NSNumber]] = ["image": imageMultiArr,
                                                       "output_true": outputMultiArr]
        
        dictionaryArray.append(dataPointFeatures)

//        if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
//            featureProviders.append(provider)
//        }
    }

//    return featureProviders
    return dictionaryArray
}


print("Start")

//let (xTrain, yTrain) = processFileOptimized(path: URL(fileURLWithPath: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_train.csv"))
//print(xTrain.count, yTrain.count)
//
//let (xTest, yTest) = processFileOptimized(path: URL(fileURLWithPath: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_test.csv"))
//print(xTest.count, yTest.count)

let batch = prepareTrainingBatch(path: URL(fileURLWithPath: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_train.csv"))

print(batch.count)

do {
    try (batch as NSArray).write(to: URL(fileURLWithPath: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_train.plist"))
}
catch {
    print("Unexpected error: \(error).")
}

//let batchProvider = MLArrayBatchProvider(array: batch)
//print(batchProvider.count)
//

