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
    @State var prediction = "-"
  
    let splitRatio: CGFloat = 0.2445
//  original: let splitRatio: CGFloat = 0.2445
    
    public func isDataReady(for status: MNIST.BatchPreparationStatus) -> Bool {
        switch status {
        case .ready: return true
        default: return false
        }
    }

    public func isDataPreparing(for status: MNIST.BatchPreparationStatus) -> Bool {
        switch status {
        case .preparing: return true
        default: return false
        }
    }

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                Form {
                    Section(header: Text("Dataset")) {
                        HStack {
                            Text("Training Size")
                            Spacer()
                            if self.isDataReady(for: self.mnist.trainingBatchStatus) {
                                Text("\(String(self.mnist.getTrainFileSize()/1000000) + "MB")")
                            } else {
                                Text("n/a")
                            }
                        }
                        HStack {
                            Text("Validation Size")
                            Spacer()
                            if self.isDataReady(for: self.mnist.predictionBatchStatus) {
                                Text("\(String(self.mnist.getValidFileSize()/1000000) + "MB")")
                            } else {
                                Text("n/a")
                            }
                        }
                        HStack {
                            Text("Remove MNIST CSV Files")
                            Spacer()
                            Button(action: {self.mnist.removeMNISTData()}) {
                                Text("Cleanup")
                            }
                        }
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
                            Text("Validation: \(self.mnist.predictionBatchStatus.description)")
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
                        Stepper(value: self.$mnist.epoch, in: 1...30, label: { Text("Epoch:  \(self.mnist.epoch)")})
                        Stepper(value: self.$mnist.hiddenNeurons, in: 50...750, step: 25, label: {
                            Text("Neurons: \(self.mnist.hiddenNeurons)")
                        })
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
                                self.prediction = "-"
                                self.drawData.lines.removeAll()
                            }
                            Spacer()
                            Text(self.prediction)
                            Spacer()
                            Button(action: {}) {
                                Text("Predict")
                            }
                                .disabled(!self.mnist.modelTrained)
                                .onTapGesture {
                                    let data = self.drawData.view.getImageData()
                                    self.prediction = "\(self.mnist.predict(data: data))"
                                }
                        }
                    }
                }.frame(width: geometry.size.width, height: geometry.size.height - (geometry.size.height * self.splitRatio))
                
                Draw()
                    .environmentObject(self.drawData)
                    .frame(width: geometry.size.height * self.splitRatio, height: geometry.size.height * self.splitRatio)
                    .border(Color.blue, width: 1)
            }
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
