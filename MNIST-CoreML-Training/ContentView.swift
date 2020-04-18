//
//  ContentView.swift
//  MNIST-CoreML-Training
//
//  Created by Jacopo Mangiavacchi on 3/29/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var mnist = MNIST()
    
    func isDataReady(for status: MNIST.BatchPreparationStatus) -> Bool {
        switch status {
        case .ready: return true
        default: return false
        }
    }

    func isDataPreparing(for status: MNIST.BatchPreparationStatus) -> Bool {
        switch status {
        case .preparing: return true
        default: return false
        }
    }

    var body: some View {
        Form {
            Section(header: Text("Dataset")) {
                HStack {
                    Text("Training: \(mnist.trainingBatchStatus.description)")
                    if isDataReady(for: mnist.trainingBatchStatus) {
                        Text(" \(mnist.trainingBatchProvider!.count) samples")
                    }
                    Spacer()
                    Button(action: {
                        self.mnist.asyncPrepareTrainBatchProvider()
                    }) {
                        Text("Start")
                    }.disabled(isDataPreparing(for: mnist.trainingBatchStatus))
                }
                HStack {
                    Text("Prediction: \(mnist.predictionBatchStatus.description)")
                    if isDataReady(for: mnist.predictionBatchStatus) {
                        Text(" \(mnist.predictionBatchProvider!.count) samples")
                    }
                    Spacer()
                    Button(action: {
                        self.mnist.asyncPreparePredictionBatchProvider()
                    }) {
                        Text("Start")
                    }.disabled(isDataPreparing(for: mnist.predictionBatchStatus))
                }
            }
            Section(header: Text("Training")) {
                Stepper(value: $mnist.epoch, in: 1...10, label: { Text("Epoch:  \(mnist.epoch)")})
                HStack {
                    Text("Prepare model")
                    Spacer()
                    Button(action: {
                        self.mnist.prepareModel()
                    }) {
                        Text("Start")
                    }.disabled(!isDataReady(for: mnist.trainingBatchStatus))
                }
                HStack {
                    Text("Compile model")
                    Spacer()
                    Button(action: {
                        self.mnist.compileModel()
                    }) {
                        Text("Start")
                    }.disabled(!mnist.modelPrepared)
                }
                HStack {
                    Text(self.mnist.modelStatus)
                    Spacer()
                    Button(action: {
                        self.mnist.trainModel()
                    }) {
                        Text("Start")
                    }.disabled(!mnist.modelCompiled)
                }
            }
            Section(header: Text("Prediction")) {
                HStack {
                    Text("Predict Test data")
                    Spacer()
                    Button(action: {
                        self.mnist.testModel()
                    }) {
                        Text("Start")
                    }.disabled(!isDataReady(for: mnist.predictionBatchStatus) || !mnist.modelTrained)
                }
                Text(self.mnist.accuracy)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
