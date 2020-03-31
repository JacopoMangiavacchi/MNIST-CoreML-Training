//
//  MNIST.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 3/31/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

public class MNIST : ObservableObject {
    public enum BatchPreparationStatus {
        case notPrepared
        case preparing(count: Int)
        case ready
        
        var description: String {
            switch self {
            case .notPrepared:
                return "Not Prepared"
            case .preparing(let count):
                return "Preparing \(count)"
            case .ready:
                return "Ready"
            }
        }
    }
    
    @Published public var batchProvider: MLBatchProvider?
    @Published public var batchStatus = BatchPreparationStatus.notPrepared
    
    public init() {
    }
    
    func asyncPrepareBatchProvider() {
        func prepareBatchProvider() -> MLBatchProvider {
            func oneHotEncode(_ n: Int) -> [Int] {
                var encode = Array(repeating: 0, count: 10)
                encode[n] = 1
                return encode
            }

            var featureProviders = [MLFeatureProvider]()
            
            var count = 0
            errno = 0
            let trainFilePath = Bundle.main.url(forResource: "mnist_train", withExtension: "csv")!
            if freopen(trainFilePath.path, "r", stdin) == nil {
                print("error opening file")
            }
            while let line = readLine()?.split(separator: ",") {
                count += 1
                DispatchQueue.main.async {
                    self.batchStatus = .preparing(count: count)
                }

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
        
        self.batchStatus = .preparing(count: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            let provider = prepareBatchProvider()
            DispatchQueue.main.async {
                self.batchProvider = provider
                self.batchStatus = .ready
            }
        }
    }
}
