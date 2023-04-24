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
        NavigationView {
            GeometryReader { geometry in
                VStack(spacing: 0) {
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
                            Stepper(value: self.$mnist.epoch, in: 1...10, label: { Text("Training epochs:  \(self.mnist.epoch)")})
                            HStack {
                                Text("Build model")
                                Spacer()
                                Button(action: {
                                    self.mnist.prepareModel()
                                    self.mnist.compileModel()
                                }) {
                                    Text("Start")
                                }.disabled(!self.isDataReady(for: self.mnist.trainingBatchStatus))
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
                                Text("Compute accuracy")
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
                    
                    Spacer(minLength: 15)
                    Draw()
                        .environmentObject(self.drawData)
                        .frame(width: geometry.size.height * self.splitRatio, height: geometry.size.height * self.splitRatio)
                        .cornerRadius(20)
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.blue, lineWidth:5))
                }
            }
            .navigationTitle("MNIST CoreML")
        }
        .navigationViewStyle(.stack)
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
