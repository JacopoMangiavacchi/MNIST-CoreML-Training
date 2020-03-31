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

    var coreMLModelUrl: URL
    
    public init() {
        coreMLModelUrl = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MNIST_Model")
            .appendingPathExtension("mlmodel")
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
                                shortDescription: "MNIST-Trainable",
                                author: "Jacopo Mangiavacchi",
                                license: "MIT",
                                userDefined: ["SwiftCoremltoolsVersion" : "0.0.8"]) {
            Input(name: "image", shape: [28, 28])
            Output(name: "output", shape: [10])
            TrainingInput(name: "image", shape: [28, 28])
            TrainingInput(name: "output_true", shape: [10])
            NeuralNetwork(losses: [CategoricalCrossEntropy(name: "lossLayer",
                                       input: "output",
                                       target: "output_true")],
                          optimizer: Adam(learningRateDefault: 0.001,
                                         learningRateMax: 0.3,
                                         miniBatchSizeDefault: 32,
                                         miniBatchSizeRange: [32],
                                         beta1Default: 0.9,
                                         beta1Max: 1.0,
                                         beta2Default: 0.999,
                                         beta2Max: 1.0,
                                         epsDefault: 0.00000001,
                                         epsMax: 0.00000001),
                          epochDefault: 6,
                          epochSet: [6],
                          shuffle: true) {
                Convolution(name: "convolution1",
                             input: ["image"],
                             output: ["outConvolution1"],
                             outputChannels: 32,
                             kernelChannels: 1,
                             nGroups: 1,
                             kernelSize: [3, 3],
                             stride: [1, 1],
                             dilationFactor: [1, 1],
                             paddingType: .same(mode: .bottomRightHeavy),
                             outputShape: [],
                             deconvolution: false,
                             updatable: true)
//                ReLu(name: "relu1",
//                     input: ["outConvolution1"],
//                     output: ["outRelu1"])
//                Pooling(name: "pooling1",
//                             input: ["outRelu1"],
//                             output: ["outPooling1"],
//                             mode: .last)
//                Flatten(name: "flatten1",
//                             input: ["outPooling1"],
//                             output: ["outFlatten1"],
//                             mode: .last)
//                InnerProduct(name: "dense1",
//                             input: ["outFlatten1"],
//                             output: ["outDense1"],
//                             inputChannels: 288,
//                             outputChannels: 500,
//                             updatable: true)
//                ReLu(name: "relu2",
//                     input: ["outDense1"],
//                     output: ["outRelu2"])
//                InnerProduct(name: "dense2",
//                             input: ["outRelu2"],
//                             output: ["outDense2"],
//                             inputChannels: 500,
//                             outputChannels: 10,
//                             updatable: true)
//                Softsign(name: "softsign",
//                     input: ["outDense2"],
//                     output: ["output"])
            }
        }
        
        let coreMLData = coremlModel.coreMLData
        print(coreMLModelUrl)
        try! coreMLData!.write(to: coreMLModelUrl)
    }
}
