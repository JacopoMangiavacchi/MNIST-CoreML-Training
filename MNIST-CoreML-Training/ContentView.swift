//
//  MNIST.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 3/29/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

public class MNIST : ObservableObject {
    @Published public var batchProvider: MLBatchProvider?
    
    public init() {
        DispatchQueue.global(qos: .userInitiated).async {
            let provider = self.prepareBatchProvider()
            DispatchQueue.main.async {
                self.batchProvider = provider
            }
        }
    }

    func prepareBatchProvider() -> MLBatchProvider {
        func oneHotEncode(_ n: Int) -> [Int] {
            var encode = Array(repeating: 0, count: 10)
            encode[n] = 1
            return encode
        }

        var featureProviders = [MLFeatureProvider]()
        
        errno = 0
        let trainFilePath = Bundle.main.url(forResource: "mnist_train", withExtension: "csv")!
        if freopen(trainFilePath.path, "r", stdin) == nil {
            print("error opening file")
        }
        while let line = readLine()?.split(separator: ",") {
            let imageMultiArr = try! MLMultiArray(shape: [28, 28], dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: [10], dataType: .int32)

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
            
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)

            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                               "output_true": outputValue]
            
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }
}
