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
    @Published public var modelPrepared = false
    @Published public var modelCompiled = false
    @Published public var modelTrained = false
    @Published public var modelStatus = "Train model"
    
    var coreMLModelUrl: URL
    var coreMLCompiledModelUrl: URL?
    var model: MLModel?
    var retrainedModel: MLModel?
    
    public init() {
        coreMLModelUrl = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MNIST_Model")
            .appendingPathExtension("mlmodel")
    }
    
    public func asyncPrepareBatchProvider() {
        func prepareBatchProvider() -> MLBatchProvider {
            func oneHotEncode(_ n: Int) -> [Float] {
                var encode = Array(repeating: Float(0), count: 10)
                encode[n] = Float(1)
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

                let imageMultiArr = try! MLMultiArray(shape: [1, 28, 28], dataType: .float32)
                let outputMultiArr = try! MLMultiArray(shape: [10], dataType: .float32)

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
    
    public func prepareModel() {
        let coremlModel = Model(version: 4,
                                shortDescription: "MNIST-Trainable",
                                author: "Jacopo Mangiavacchi",
                                license: "MIT",
                                userDefined: ["SwiftCoremltoolsVersion" : "0.0.11"]) {
            Input(name: "image", shape: [1, 28, 28])
            Output(name: "output", shape: [10])
            TrainingInput(name: "image", shape: [1, 28, 28])
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
                Convolution(name: "conv1",
                             input: ["image"],
                             output: ["outConv1"],
                             outputChannels: 32,
                             kernelChannels: 1,
                             nGroups: 1,
                             kernelSize: [3, 3],
                             stride: [1, 1],
                             dilationFactor: [1, 1],
                             paddingType: .same(mode: .bottomRightHeavy),
                             outputShape: [],
                             deconvolution: false,
                             updatable: false)
                ReLu(name: "relu1",
                     input: ["outConv1"],
                     output: ["outRelu1"])
                Pooling(name: "pooling1",
                             input: ["outRelu1"],
                             output: ["outPooling1"],
                             poolingType: .max,
                             kernelSize: [2, 2],
                             stride: [2, 2],
                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
                             avgPoolExcludePadding: true,
                             globalPooling: false)
                Convolution(name: "conv2",
                             input: ["outPooling1"],
                             output: ["outConv2"],
                             outputChannels: 32,
                             kernelChannels: 32,
                             nGroups: 1,
                             kernelSize: [2, 2],
                             stride: [1, 1],
                             dilationFactor: [1, 1],
                             paddingType: .same(mode: .bottomRightHeavy),
                             outputShape: [],
                             deconvolution: false,
                             updatable: false)
                ReLu(name: "relu2",
                     input: ["outConv2"],
                     output: ["outRelu2"])
                Pooling(name: "pooling2",
                             input: ["outRelu2"],
                             output: ["outPooling2"],
                             poolingType: .max,
                             kernelSize: [2, 2],
                             stride: [2, 2],
                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
                             avgPoolExcludePadding: true,
                             globalPooling: false)
                Convolution(name: "conv3",
                             input: ["outPooling2"],
                             output: ["outConv3"],
                             outputChannels: 32,
                             kernelChannels: 32,
                             nGroups: 1,
                             kernelSize: [2, 2],
                             stride: [1, 1],
                             dilationFactor: [1, 1],
                             paddingType: .same(mode: .bottomRightHeavy),
                             outputShape: [],
                             deconvolution: false,
                             updatable: false)
                ReLu(name: "relu3",
                     input: ["outConv3"],
                     output: ["outRelu3"])
                Pooling(name: "pooling3",
                             input: ["outRelu3"],
                             output: ["outPooling3"],
                             poolingType: .max,
                             kernelSize: [2, 2],
                             stride: [2, 2],
                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
                             avgPoolExcludePadding: true,
                             globalPooling: false)
                Flatten(name: "flatten1",
                             input: ["outPooling3"],
                             output: ["outFlatten1"],
                             mode: .last)
                InnerProduct(name: "hidden1",
                             input: ["outFlatten1"],
                             output: ["outHidden1"],
                             inputChannels: 288,
                             outputChannels: 500,
                             updatable: true)
                ReLu(name: "relu4",
                     input: ["outHidden1"],
                     output: ["outRelu4"])
                InnerProduct(name: "hidden2",
                             input: ["outRelu4"],
                             output: ["outHidden2"],
                             inputChannels: 500,
                             outputChannels: 10,
                             updatable: true)
                Softmax(name: "softmax",
                        input: ["outHidden2"],
                        output: ["output"])
            }
        }
        
        let coreMLData = coremlModel.coreMLData
        print(coreMLModelUrl)
        try! coreMLData!.write(to: coreMLModelUrl)
        modelPrepared = true
    }
    
    public func compileModel() {
        coreMLCompiledModelUrl = try! MLModel.compileModel(at: coreMLModelUrl)
        print("Compiled Model Path: \(coreMLCompiledModelUrl!)")
        model = try! MLModel(contentsOf: coreMLCompiledModelUrl!)
        modelCompiled = true
    }
    
    public func trainModel() {
        self.modelTrained = false
        self.modelStatus = "Training starting"
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        //configuration.parameters = [.epochs : 100]
        let progressHandler = { (context: MLUpdateContext) in
            switch context.event {
            case .trainingBegin:
                print("Training begin")
                DispatchQueue.main.async {
                    self.modelStatus = "Training begin"
                }

            case .miniBatchEnd:
                break
//                let batchIndex = context.metrics[.miniBatchIndex] as! Int
//                let batchLoss = context.metrics[.lossValue] as! Double
//                print("Mini batch \(batchIndex), loss: \(batchLoss)")
            case .epochEnd:
                let epochIndex = context.metrics[.epochIndex] as! Int
                let trainLoss = context.metrics[.lossValue] as! Double
                print("Epoch \(epochIndex) end with loss \(trainLoss)")
                DispatchQueue.main.async {
                    self.modelStatus = "Epoch \(epochIndex) end with loss \(trainLoss)"
                }

            default:
                print("Unknown event")
            }

//        print(context.model.modelDescription.parameterDescriptionsByKey)
//        do {
//            let multiArray = try context.model.parameterValue(for: MLParameterKey.weights.scoped(to: "dense_1")) as! MLMultiArray
//            print(multiArray.shape)
//        } catch {
//            print(error)
//        }
        }

        let completionHandler = { (context: MLUpdateContext) in
            print("Training completed with state \(context.task.state.rawValue)")
            print("CoreML Error: \(context.task.error.debugDescription)")
            DispatchQueue.main.async {
                self.modelStatus = "Training completed with state \(context.task.state.rawValue)"
            }

            if context.task.state != .completed {
                print("Failed")
                DispatchQueue.main.async {
                    self.modelStatus = "Training Failed"
                }
                return
            }

            let trainLoss = context.metrics[.lossValue] as! Double
            print("Final loss: \(trainLoss)")
            DispatchQueue.main.async {
                self.modelStatus = "Training completed with loss: \(trainLoss)"
                self.modelTrained = true
            }

            self.retrainedModel = context.model

//            let updatedModel = context.model
//            let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
//            try! updatedModel.write(to: updatedModelURL)
            print("Model Trained!")
        }

        let handlers = MLUpdateProgressHandlers(
                            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
                            progressHandler: progressHandler,
                            completionHandler: completionHandler)


        let updateTask = try! MLUpdateTask(forModelAt: coreMLCompiledModelUrl!,
                                           trainingData: batchProvider!,
                                           configuration: configuration,
                                           progressHandlers: handlers)

        updateTask.resume()
    }
}

