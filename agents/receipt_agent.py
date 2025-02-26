# agents/receipt_agent.py
import asyncio
import io
import logging
import os
import traceback
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

class ReceiptAnalysisAgent:
    async def process_image_async(self, image_bytes: bytes) -> str:
        try:
            # Get Azure Form Recognizer configuration from environment variables.
            endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
            key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
            if not endpoint or not key:
                logger.error("Azure Form Recognizer endpoint or key not set in environment variables.")
                return "Azure configuration error."

            # Create the Form Recognizer client.
            document_analysis_client = DocumentAnalysisClient(
                endpoint=endpoint, 
                credential=AzureKeyCredential(key)
            )
            logger.info("Azure Form Recognizer client initialized.")

            async with document_analysis_client:
                # Analyze the receipt image using the prebuilt receipt model.
                poller = await document_analysis_client.begin_analyze_document(
                    "prebuilt-receipt", document=io.BytesIO(image_bytes)
                )
                result = await poller.result()

                output_lines = []
                for receipt in result.documents:
                    merchant = receipt.fields.get("MerchantName", {}).get("value", "N/A")
                    transaction_date = receipt.fields.get("TransactionDate", {}).get("value", "N/A")
                    total_amount = receipt.fields.get("Total", {}).get("value", "N/A")
                    output_lines.append(f"Merchant: {merchant}")
                    output_lines.append(f"Transaction Date: {transaction_date}")
                    output_lines.append(f"Total Amount: {total_amount}")
                    
                    items_field = receipt.fields.get("Items", {}).get("value", [])
                    if items_field:
                        output_lines.append("Items:")
                        for item in items_field:
                            description = item.get("Description", {}).get("value", "N/A")
                            item_price = item.get("TotalPrice", {}).get("value", "N/A")
                            output_lines.append(f"  - {description}: {item_price}")
                    else:
                        output_lines.append("No items detected.")
                
                extracted_text = "\n".join(output_lines)
                logger.info("Azure OCR processing completed.")
                logger.info(f"Extracted Receipt Data:\n{extracted_text}")
                return extracted_text.strip()

        except Exception as e:
            logger.error("Error processing receipt image using Azure Form Recognizer:")
            logger.error(traceback.format_exc())
            return "Error processing the receipt image using Azure."