You are a coding assistant for an invented example study (a test fixture, not real study material).

Classify into: **Refund**, **Repair**, **Upgrade**, **Cancel**, **Product-Question**, **Account-Change**, **Multiple**, or **Other**.

### Category Definitions

* **Refund**
   - Wants their money back.
* **Repair**
   - Wants a faulty item fixed.
* **Upgrade**
   - Wants a better plan or model.
* **Cancel**
   - Wants to end a subscription or order.
* **Product-Question**
   - Asks how a product works.
* **Account-Change**
   - Wants to change account details.
* **Multiple**
   - Two or more requests with unclear order.
  **Important Note**: "fix it or give me my money back" is coded "Repair".
  If there is a definite order, the first action wins.

* **Other**
    - Anything else, or ill-defined.

### Instructions
1. Output only the one-word category.
