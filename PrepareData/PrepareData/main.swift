//
//  main.swift
//  PrepareData
//
//  Created by Jacopo Mangiavacchi on 3/30/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation

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

func processFileOptimized(path: String) -> (image: [[Float]], label: [[Int]]) {
    var image = [[Float]]()
    var label = [[Int]]()

    errno = 0
    if freopen(path, "r", stdin) == nil {
        print("error opening file")
    }
    while let line = readLine()?.split(separator: ",") {
        label.append(oneHotEncode(Int(String(line[0]))!))
        image.append(Array(line[1...].map{ Float(String($0))! / Float(255.0) }))
    }

    return (image, label)
}

print("Start")

//let (xTrain, yTrain) = processFile(path: URL(fileURLWithPath: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_train.csv"))
let (xTrain, yTrain) = processFileOptimized(path: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_train.csv")
print(xTrain.count, yTrain.count)

//let (xTest, yTest) = processFile(path: URL(fileURLWithPath: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_test.csv"))
let (xTest, yTest) = processFileOptimized(path: "/Users/jacopo/MNIST-CoreML-Training/Data/mnist_test.csv")
print(xTest.count, yTest.count)
