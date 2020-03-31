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
                    Text("\(mnist.batchStatus.description)")
                    if isDataReady(for: mnist.batchStatus) {
                        Text(" \(mnist.batchProvider!.count) samples")
                    }
                    Spacer()
                    Button(action: {
                        self.mnist.asyncPrepareBatchProvider()
                    }) {
                        Text("Start")
                    }.disabled(isDataPreparing(for: mnist.batchStatus))
                }
            }
            Section(header: Text("Model")) {
                HStack {
                    Text("Prepare model")
                    Spacer()
                    Button(action: {
                        self.mnist.prepareModel()
                    }) {
                        Text("Start")
                    }.disabled(!isDataReady(for: mnist.batchStatus))
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
