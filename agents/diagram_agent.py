# agents/diagram_agent.py

import asyncio
import logging
import os
from io import BytesIO

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DiagramAnalysisAgent:
    def __init__(self, detection_model_path="dataset/runs/detect/electrical_model/weights/best.pt"):
        """
        Load the YOLOv8 model for detecting electrical symbols.
        """
        logger.info("Loading YOLOv8 detection model from %s", detection_model_path)
        try:
            # Load the YOLOv8 model using the ultralytics package
            self.detection_model = YOLO(detection_model_path)
        except Exception as e:
            logger.error("Failed to load YOLOv8 model: %s", e)
            self.detection_model = None

    async def analyze_diagram(self, image_bytes: bytes) -> str:
        """
        Analyze an electrical diagram image:
        - Decode the image.
        - Preprocess for OCR and structure analysis.
        - Extract text using Tesseract.
        - Identify components from text.
        - Analyze image structure (contours and lines).
        - Use YOLOv8 to detect electrical symbols.
        - Generate an educational report.
        """
        try:
            logger.info("Starting diagram analysis...")

            # Decode the image from bytes
            img = self.decode_image(image_bytes)
            if img is None:
                return "Error: Unable to decode the provided image."

            # Preprocess image (grayscale, threshold, edges)
            gray, thresh, edges = self.preprocess_image(img)

            # Extract text via OCR
            ocr_text = self.extract_text(gray)
            logger.info("OCR text: %s", ocr_text)

            # Identify components from OCR text using keywords
            components_text = self.identify_components(ocr_text)
            logger.info("Text components: %s", components_text)

            # Analyze diagram structure
            contours_info = self.analyze_contours(thresh)
            lines_info = self.analyze_lines(edges)
            logger.info("Structure analysis complete.")

            # Use YOLOv8 to detect electrical symbols
            detected_symbols = self.detect_electrical_symbols(img)
            logger.info("Detected symbols: %s", detected_symbols)

            # For each detected symbol, get additional information (placeholder)
            symbol_info = {}
            for symbol in detected_symbols:
                symbol_info[symbol] = self.query_knowledge_base(symbol)

            # Generate educational notes for text-based components
            educational_text = self.generate_educational_notes(components_text)
            educational_symbols = "\n".join([f"{symbol}: {info}" for symbol, info in symbol_info.items()])
            educational = educational_text + "\n\nElectrical Symbol Details:\n" + educational_symbols

            # Compile the final report
            report = (
                "Diagram Analysis Report:\n"
                "====================================\n"
                "1. OCR Extracted Text:\n" +
                (ocr_text if ocr_text else "None detected.") + "\n\n" +
                "2. Detected Electrical Components (from text):\n" +
                (", ".join(components_text) if components_text else "None") + "\n\n" +
                "3. Detected Electrical Symbols (via YOLOv8):\n" +
                (", ".join(detected_symbols) if detected_symbols else "None") + "\n\n" +
                "4. Diagram Structure Analysis:\n" +
                contours_info + "\n" + lines_info + "\n\n" +
                "5. Educational Notes:\n" +
                educational + "\n"
            )

            logger.info("Diagram analysis completed.")
            return report

        except Exception as e:
            logger.error("Error during diagram analysis: %s", e, exc_info=True)
            return "Diagram analysis encountered an error."

    def decode_image(self, image_bytes: bytes):
        """
        Convert image bytes into an OpenCV image.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Image decoding failed.")
        return img

    def preprocess_image(self, img):
        """
        Preprocess the image for OCR and structure analysis:
        - Convert to grayscale.
        - Apply adaptive thresholding.
        - Detect edges.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        return gray, thresh, edges

    def extract_text(self, gray_img) -> str:
        """
        Use Tesseract OCR to extract text from the grayscale image.
        """
        try:
            text = pytesseract.image_to_string(gray_img)
            return text.strip()
        except Exception as e:
            logger.error("OCR extraction error: %s", e)
            return ""

    def identify_components(self, ocr_text: str) -> list:
        """
        Identify electrical components from OCR text using keywords.
        """
        components = []
        keywords = {
            "transformer": "Transformer",
            "circuit breaker": "Circuit Breaker",
            "relay": "Relay",
            "switch": "Switch",
            "capacitor": "Capacitor",
            "inductor": "Inductor",
            "resistor": "Resistor",
            "insulator": "Insulator",
            "bus": "Bus Bar",
            "meter": "Meter",
            "disconnect": "Disconnect Switch",
        }
        text_lower = ocr_text.lower()
        for key, name in keywords.items():
            if key in text_lower:
                components.append(name)
        return components

    def analyze_contours(self, thresh_img) -> str:
        """
        Analyze contours in the thresholded image to estimate diagram complexity.
        """
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
        description = f"Detected {count} contour{'s' if count != 1 else ''}."
        if count == 0:
            description += " No clear boundaries detected."
        elif count < 5:
            description += " The diagram looks clean."
        else:
            description += " The diagram might be complex."
        return description

    def analyze_lines(self, edges_img) -> str:
        """
        Use Hough Transform to detect line segments in the image.
        """
        lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            count = len(lines)
            return f"Detected {count} line segment{'s' if count != 1 else ''}."
        else:
            return "No significant lines detected."

    def detect_electrical_symbols(self, img) -> list:
        """
        Use the YOLOv8 model to detect electrical symbols in the image.
        """
        if self.detection_model is None:
            logger.error("Detection model is not loaded.")
            return []
        # YOLOv8 expects an RGB image.
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detection_model.predict(rgb_img, verbose=False)
        detected_symbols = []
        try:
            # Iterate over results for the image
            for result in results:
                # Each result contains detected boxes.
                boxes = result.boxes
                if boxes is None or boxes.shape[0] == 0:
                    continue
                # For each detected box, get the class id and map it to a name.
                for cls in boxes.cls:
                    class_id = int(cls.cpu().numpy())
                    symbol_name = self.detection_model.model.names[class_id]
                    if symbol_name not in detected_symbols:
                        detected_symbols.append(symbol_name)
            return detected_symbols
        except Exception as e:
            logger.error("Error processing YOLOv8 detection results: %s", e)
            return []

    def query_knowledge_base(self, symbol: str) -> str:
        """
        Placeholder: Retrieve information for the given symbol.
        Replace this with your actual integration (e.g., query your Pinecone database or chatbot).
        """
        return f"Manual and troubleshooting info for {symbol}."

    def generate_educational_notes(self, components_text: list) -> str:
        """
        Provide educational notes based on the detected text components.
        """
        notes = []
        component_info = {
            "Transformer": "A transformer transfers electrical energy between circuits via electromagnetic induction.",
            "Circuit Breaker": "Circuit breakers protect circuits by interrupting power during overloads or faults.",
            "Relay": "Relays allow a small signal to control a larger electrical load.",
            "Switch": "Switches control the flow of electrical current.",
            "Capacitor": "Capacitors store electrical energy temporarily and help filter signals.",
            "Inductor": "Inductors store energy in a magnetic field and filter electrical noise.",
            "Resistor": "Resistors limit current and adjust voltage levels.",
            "Insulator": "Insulators prevent the flow of electricity and protect users from shocks.",
            "Bus Bar": "Bus bars distribute power within an electrical system.",
            "Meter": "Meters measure electrical quantities like voltage, current, and resistance.",
            "Disconnect Switch": "Disconnect switches safely isolate parts of an electrical system.",
        }
        for comp in components_text:
            note = component_info.get(comp, f"No info available for {comp}.")
            notes.append(f"{comp}: {note}")
        if not notes:
            notes.append("No electrical components detected for educational notes.")
        return "\n".join(notes)


# For testing: run this script directly by providing the path to an image.
if __name__ == "__main__":
    import sys

    async def main(image_path: str):
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            logger.error("Failed to open image file: %s", e)
            return

        agent = DiagramAnalysisAgent(detection_model_path="dataset/runs/detect/electrical_model/weights/best.pt")
        report = await agent.analyze_diagram(image_bytes)
        print(report)

    if len(sys.argv) != 2:
        print("Usage: python diagram_agent.py <path_to_diagram_image>")
    else:
        asyncio.run(main(sys.argv[1]))