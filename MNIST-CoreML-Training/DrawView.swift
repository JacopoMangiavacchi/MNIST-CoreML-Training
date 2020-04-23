//
//  DrawView.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 4/21/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//
// Code from Apple's Metal-2 sample MPSCNNHelloWorld

import UIKit
import SwiftUI

/**
 This class is used to handle the drawing in the DigitView so we can get user input digit,
 This class doesn't really have an MPS or Metal going in it, it is just used to get user input
 */
class DrawView: UIView {
    
    // some parameters of how thick a line to draw 15 seems to work
    // and we have white drawings on black background just like MNIST needs its input
    var linewidth = CGFloat(15) { didSet { setNeedsDisplay() } }
    var color = UIColor.white { didSet { setNeedsDisplay() } }
    
    // we will keep touches made by user in view in these as a record so we can draw them.
    var lines: DrawData!
    var lastPoint: CGPoint!
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        lastPoint = touches.first!.location(in: self)
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        let newPoint = touches.first!.location(in: self)
        // keep all lines drawn by user as touch in record so we can draw them in view
        lines.lines.append(Line(start: lastPoint, end: newPoint))
        lastPoint = newPoint
        // make a draw call
        setNeedsDisplay()
    }
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        let drawPath = UIBezierPath()
        drawPath.lineCapStyle = .round
        
        for line in lines.lines {
            drawPath.move(to: line.start)
            drawPath.addLine(to: line.end)
        }
        
        drawPath.lineWidth = linewidth
        color.set()
        drawPath.stroke()
    }
    
    
    /**
     This function gets the pixel data of the view so we can put it in MTLTexture
     
     - Returns:
     Void
     */
    func getViewContext() -> CGContext? {
        // our network takes in only grayscale images as input
        let colorSpace:CGColorSpace = CGColorSpaceCreateDeviceGray()
        
        // we have 3 channels no alpha value put in the network
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        
        // this is where our view pixel data will go in once we make the render call
        let context = CGContext(data: nil, width: 28, height: 28, bitsPerComponent: 8, bytesPerRow: 28, space: colorSpace, bitmapInfo: bitmapInfo)
        
        // scale and translate so we have the full digit and in MNIST standard size 28x28
        context!.translateBy(x: 0 , y: 28)
        context!.scaleBy(x: 28/self.frame.size.width, y: -28/self.frame.size.height)
        
        // put view pixel data in context
        self.layer.render(in: context!)
        
        return context
    }
    
    func getImageData() -> [[Float]] {
        let cgImage = getViewContext()?.makeImage()
        let bitmap = Bitmap(img: cgImage!)
        
        var pixelArray: [[Float]] = Array(repeating: Array(repeating: 0.0, count: 28), count: 28)
        for row in 0..<28{
            for col in 0..<28 {
                pixelArray[row][col] = Float(bitmap.color_at(x: col, y: row).0) / 255.0
            }
        }

        return pixelArray
    }
}

class Bitmap {
    let width: Int
    let height: Int
    let context: CGContext

    init(img: CGImage) {
        // Set image width, height
        width = img.width
        height = img.height

        // Declare the number of bytes per row. Each pixel in the bitmap in this
        // example is represented by 4 bytes; 8 bits each of red, green, blue, and
        // alpha.
        let bitmapBytesPerRow = width * 4

        // Use the generic RGB color space.
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)

        // Create the bitmap context. We want pre-multiplied ARGB, 8-bits
        // per component. Regardless of what the source image format is
        // (CMYK, Grayscale, and so on) it will be converted over to the format
        // specified here by CGBitmapContextCreate.
        context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bitmapBytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)!

        // draw the image onto the context
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        context.draw(img, in: rect)
    }

    func color_at(x: Int, y: Int) -> (Int, Int, Int, Int) {
        assert(0<=x && x<width)
        assert(0<=y && y<height)
        
        let offset = 4 * (y * width + x)
        var data = context.data!.advanced(by: offset)
        
        let alpha = data.load(as: UInt8.self)
        data = data.advanced(by: 1)
        let red = data.load(as: UInt8.self)
        data = data.advanced(by: 1)
        let green = data.load(as: UInt8.self)
        data = data.advanced(by: 1)
        let blue = data.load(as: UInt8.self)
        data = data.advanced(by: 1)

        let color = (Int(red), Int(green), Int(blue), Int(alpha))
        return color
    }
}

class Line{
    var start, end: CGPoint
    
    init(start: CGPoint, end: CGPoint) {
        self.start = start
        self.end   = end
    }
}

struct Draw: UIViewRepresentable {
    typealias UIViewType = DrawView
    
    @EnvironmentObject var drawData: DrawData

    func makeUIView(context: Context) -> DrawView {
        return drawData.view
    }

    func updateUIView(_ uiView: DrawView, context: UIViewRepresentableContext<Draw>) {
        uiView.setNeedsDisplay()
    }
}

class DrawData: ObservableObject {
    @Published var lines = [Line]()
    var view: DrawView
    
    init() {
        self.view = DrawView()
        self.view.lines = self
        self.view.backgroundColor = .black
    }
}
