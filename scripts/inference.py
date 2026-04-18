import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class IntentClassification:
    def __init__(self, model_path="Ganth1811/banking-intent-llama-3.2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"loading model: {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,         
            torch_dtype=dtype, 
            low_cpu_mem_usage=True
        )
        
        self.model.tie_weights()
        self.model.eval()
        
        
        self.prompt_template = """Here is a banking query:
{}

Classify this query into one of the following intents, respond only
with a number depicting that class.

class 0: activate_my_card
class 1: age_limit
class 2: apple_pay_or_google_pay
class 3: atm_supportuô
class 4: automatic_top_up
class 5: balance_not_updated_after_bank_transfer
class 6: balance_not_updated_after_cheque_or_cash_deposit
class 7: beneficiary_not_allowed
class 8: cancel_transfer
class 9: card_about_to_expire
class 10: card_acceptance
class 11: card_arrival
class 12: card_delivery_estimate
class 13: card_linking
class 14: card_not_working
class 15: card_payment_fee_charged
class 16: card_payment_not_recognised
class 17: card_payment_wrong_exchange_rate
class 18: card_swallowed
class 19: cash_withdrawal_charge
class 20: cash_withdrawal_not_recognised
class 21: change_pin
class 22: compromised_card
class 23: contactless_not_working
class 24: country_support
class 25: declined_card_payment
class 26: declined_cash_withdrawal
class 27: declined_transfer
class 28: direct_debit_payment_not_recognised
class 29: disposable_card_limits
class 30: edit_personal_details
class 31: exchange_charge
class 32: exchange_rate
class 33: exchange_via_app
class 34: extra_charge_on_statement
class 35: failed_transfer
class 36: fiat_currency_support
class 37: get_disposable_virtual_card
class 38: get_physical_card
class 39: getting_spare_card
class 40: getting_virtual_card
class 41: lost_or_stolen_card
class 42: lost_or_stolen_phone
class 43: order_physical_card
class 44: passcode_forgotten
class 45: pending_card_payment
class 46: pending_cash_withdrawal
class 47: pending_top_up
class 48: pending_transfer
class 49: pin_blocked
class 50: receiving_money
class 51: Refund_not_showing_up
class 52: request_refund
class 53: reverted_card_payment?
class 54: supported_cards_and_currencies
class 55: terminate_account
class 56: top_up_by_bank_transfer_charge
class 57: top_up_by_card_charge
class 58: top_up_by_cash_or_cheque
class 59: top_up_failed
class 60: top_up_limits
class 61: top_up_reverted
class 62: topping_up_by_card
class 63: transaction_charged_twice
class 64: transfer_fee_charged
class 65: transfer_into_account
class 66: transfer_not_received_by_recipient
class 67: transfer_timing
class 68: unable_to_verify_identity
class 69: verify_my_identity
class 70: verify_source_of_funds
class 71: verify_top_up
class 72: virtual_card_not_working
class 73: visa_or_mastercard
class 74: why_verify_identity
class 75: wrong_amount_of_cash_received
class 76: wrong_exchange_rate_for_cash_withdrawal

SOLUTION
The correct answer is: class """

        self.valid_classes = []
        self.number_token_ids = []
        self._prepare_token_ids()
        label_path = "configs/label_mapping.json"
        self.label_map = {}
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            print(f"Loaded {len(self.label_map)} labels from {label_path}")
        except FileNotFoundError:
            print(f"Warning: Label file not found at {label_path}. Will return only IDs.")
        
        print("Inference is ready")

    def _prepare_token_ids(self):
        for i in range(77):
            tokens = self.tokenizer.encode(str(i), add_special_tokens=False)
            if len(tokens) > 0:
                self.number_token_ids.append(tokens[0])
                self.valid_classes.append(i)

    def __call__(self, message):
        if not message or not message.strip():
            raise ValueError("Question cannot be empty")

        prompt = self.prompt_template.format(message.strip())

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_token_logits = outputs.logits[0, -1, :]
            target_logits = last_token_logits[self.number_token_ids]
            best_index = torch.argmax(target_logits).item()
            predicted_label = self.valid_classes[best_index]
            
            predicted_name = self.label_map.get(str(predicted_label), "Unknown Label")
        return predicted_label, predicted_name


if __name__ == "__main__":
    try:
        classifier = IntentClassification()
        print("\nSystem Ready. Type 'exit' to quit.")
        
        while True:
            query = input("\n💬 User Query: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            
            if not query.strip(): continue
            
            start = time.time()
            label, label_name = classifier(query)
            end = time.time()
            
            print(f"🤖 Predicted Class: {label}")
            print(f"🏷️ Label Name: {label_name}")
            print(f"⏱️ Speed: {end - start:.2f}s")
            
    except KeyboardInterrupt:
        print("\n👋 System stopped.")