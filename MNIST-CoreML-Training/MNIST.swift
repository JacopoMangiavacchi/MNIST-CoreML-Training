//
//  MNIST.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 3/31/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML
import SwiftCoreMLTools

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
    
    func prepareModel() {
        let coremlModel = Model(version: 4,
                                shortDescription: "Regression",
                                author: "Jacopo Mangiavacchi",
                                license: "MIT",
                                userDefined: ["SwiftCoremltoolsVersion" : "0.0.6"]) {
            Input(name: "numericalInput", shape: [11])
            Input(name: "categoricalInput1", shape: [1])
            Input(name: "categoricalInput2", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "numericalInput", shape: [11])
            TrainingInput(name: "categoricalInput1", shape: [1])
            TrainingInput(name: "categoricalInput2", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
            NeuralNetwork(losses: [MSE(name: "lossLayer",
                                       input: "output",
                                       target: "output_true")],
                          optimizer: SGD(learningRateDefault: 0.001,
                                         learningRateMax: 0.3,
                                         miniBatchSizeDefault: 32,
                                         miniBatchSizeRange: [32],
                                         momentumDefault: 0,
                                         momentumMax: 1.0),
                          epochDefault: 500,
                          epochSet: [500],
                          shuffle: true) {
                Embedding(name: "embedding1",
                             input: ["categoricalInput1"],
                             output: ["outEmbedding1"],
                             weight: [Float](),
                             inputDim: 2,
                             outputChannels: 2)
                Permute(name: "permute1",
                             input: ["outEmbedding1"],
                             output: ["outPermute1"],
                             axis: [2, 1, 0, 3])
                Flatten(name: "flatten1",
                             input: ["outPermute1"],
                             output: ["outFlatten1"],
                             mode: .last)
                Embedding(name: "embedding2",
                             input: ["categoricalInput2"],
                             output: ["outEmbedding2"],
                             weight: [Float](),
                             inputDim: 9,
                             outputChannels: 5)
                Permute(name: "permute2",
                             input: ["outEmbedding2"],
                             output: ["outPermute2"],
                             axis: [2, 1, 0, 3])
                Flatten(name: "flatten2",
                             input: ["outPermute2"],
                             output: ["outFlatten2"],
                             mode: .last)
                Concat(name: "concat",
                             input: ["numericalInput", "outFlatten1", "outFlatten2"],
                             output: ["outConcat"])
                InnerProduct(name: "dense1",
                             input: ["outConcat"],
                             output: ["outDense1"],
                             inputChannels: 11 + 2 + 5,
                             outputChannels: 64,
                             updatable: true)
                ReLu(name: "Relu1",
                     input: ["outDense1"],
                     output: ["outRelu1"])
                InnerProduct(name: "dense2",
                             input: ["outRelu1"],
                             output: ["outDense2"],
                             inputChannels: 64,
                             outputChannels: 32,
                             updatable: true)
                ReLu(name: "Relu2",
                     input: ["outDense2"],
                     output: ["outRelu2"])
                InnerProduct(name: "dense3",
                             input: ["outRelu2"],
                             output: ["output"],
                             inputChannels: 32,
                             outputChannels: 1,
                             updatable: true)
            }
        }
        
        let coreMLData = coremlModel.coreMLData
        
        let contentURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MNIST_Model")
            .appendingPathExtension("mlmodel")
        
        print(contentURL)
        
        try! coreMLData!.write(to: contentURL)
    }
}
