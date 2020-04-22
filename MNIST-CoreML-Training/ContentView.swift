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
    @ObservedObject var drawData = DrawData()
    
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
        VStack {
            Form {
                Section(header: Text("Dataset")) {
                    HStack {
                        Text("Training: \(self.mnist.trainingBatchStatus.description)")
                        if self.isDataReady(for: self.mnist.trainingBatchStatus) {
                            Text(" \(self.mnist.trainingBatchProvider!.count) samples")
                        }
                        Spacer()
                        Button(action: {
                            self.mnist.asyncPrepareTrainBatchProvider()
                        }) {
                            Text("Start")
                        }.disabled(self.isDataPreparing(for: self.mnist.trainingBatchStatus))
                    }
                    HStack {
                        Text("Prediction: \(self.mnist.predictionBatchStatus.description)")
                        if self.isDataReady(for: self.mnist.predictionBatchStatus) {
                            Text(" \(self.mnist.predictionBatchProvider!.count) samples")
                        }
                        Spacer()
                        Button(action: {
                            self.mnist.asyncPreparePredictionBatchProvider()
                        }) {
                            Text("Start")
                        }.disabled(self.isDataPreparing(for: self.mnist.predictionBatchStatus))
                    }
                }
                Section(header: Text("Training")) {
                    Stepper(value: self.$mnist.epoch, in: 1...10, label: { Text("Epoch:  \(self.mnist.epoch)")})
                    HStack {
                        Text("Prepare model")
                        Spacer()
                        Button(action: {
                            self.mnist.prepareModel()
                        }) {
                            Text("Start")
                        }.disabled(!self.isDataReady(for: self.mnist.trainingBatchStatus))
                    }
                    HStack {
                        Text("Compile model")
                        Spacer()
                        Button(action: {
                            self.mnist.compileModel()
                        }) {
                            Text("Start")
                        }.disabled(!self.mnist.modelPrepared)
                    }
                    HStack {
                        Text(self.mnist.modelStatus)
                        Spacer()
                        Button(action: {
                            self.mnist.trainModel()
                        }) {
                            Text("Start")
                        }.disabled(!self.mnist.modelCompiled)
                    }
                }
                Section(header: Text("Validation")) {
                    HStack {
                        Text("Predict Test data")
                        Spacer()
                        Button(action: {
                            self.mnist.testModel()
                        }) {
                            Text("Start")
                        }.disabled(!self.isDataReady(for: self.mnist.predictionBatchStatus) || !self.mnist.modelTrained)
                    }
                    Text(self.mnist.accuracy)
                }
                Section(header: Text("Test")) {
                    HStack {
                        Button(action: {}) {
                            Text("Clear")
                        }.onTapGesture {
                            self.drawData.lines.removeAll()
                        }
                        Spacer()
                        Text("-")
                        Spacer()
                        Button(action: {}) {
                            Text("Detect")
                        }.onTapGesture {
                            let pb = self.drawData.view.getPixelBuffer()
                            print(pb)
                        }
                    }
                }
            }
            Draw()
                .environmentObject(self.drawData)
                .frame(minWidth: 200, maxWidth: 200, minHeight: 200, maxHeight: 200)
                .border(Color.blue, width: 1)
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
