//
//  ViewController.swift
//  Tumor Classifier
//
//  Created by Jakub Stepien on 05/06/2022.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
	
	@IBAction func PickPhoto(_ sender: UIButton) {
		view.backgroundColor = .white
		presentPhotoPicker(sourceType: .photoLibrary)
	}
	
	@IBOutlet var ImageViewOutlet: UIImageView!
	
	@IBOutlet var PredictionLabel: UILabel!
	
	let tumorModel = try? TumorClassifierModel(configuration: MLModelConfiguration())
	
	override func viewDidLoad() {
		super.viewDidLoad()
		view.backgroundColor = .darkGray
	}
	
	func presentPhotoPicker(sourceType: UIImagePickerController.SourceType) {
		let picker = UIImagePickerController()
		picker.delegate = self
		picker.sourceType = sourceType
		present(picker, animated: true)
	}
	
	func pixelBuffer(for image: UIImage)-> CVPixelBuffer? {
		let model = tumorModel!.model
		
		let imageConstraint = model.modelDescription
			.inputDescriptionsByName["sequential_5_input"]!
			.imageConstraint!
		
		let imageOptions: [MLFeatureValue.ImageOption: Any] = [
			.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
		]
		
		return try? MLFeatureValue(
			cgImage: image.cgImage!,
			constraint: imageConstraint,
			options: imageOptions).imageBufferValue
	}
	
	func classify(image: UIImage) {
		DispatchQueue.global(qos: .userInitiated).async {
			
			if let pixelBuffer = self.pixelBuffer(for: image) {
				
				if let prediction = try? self.tumorModel?.prediction(sequential_5_input: pixelBuffer) {
					
					let results = self.top(4, prediction.Identity)
					self.processObservations(results: results)
				} else {
					self.processObservations(results: [])
				}
			}
		}
	}
	
	func top(_ k: Int, _ prob: [String: Double]) -> [(String, Double)] {
		return Array(prob.sorted { $0.value > $1.value }
			.prefix(min(k, prob.count)))
	}
	
	func processObservations(
		results: [(identifier: String, confidence: Double)]) {
			DispatchQueue.main.async {
				if results.isEmpty {
					self.PredictionLabel.text = "nothing found"
				} else if results[0].confidence < 0.5 {
					self.PredictionLabel.text = "not sure"
				} else {
					let top4 = results.prefix(4).map { observation in
						String(format: "%@ %.1f%%", observation.identifier,
							   observation.confidence * 100)
					}
					self.PredictionLabel.text = top4.joined(separator: "\n")
				}
			}
		}
	
}


extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
	func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
		picker.dismiss(animated: true)
		
		let image = info[.originalImage] as! UIImage
		ImageViewOutlet.image = image
		
		classify(image: image)
	}
}

