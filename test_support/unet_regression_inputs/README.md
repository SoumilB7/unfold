# Exact UNet regression inputs

These JSON files are test inputs, not production model rules. `provenance.json` pins their exact prepared bytes.

`kandinsky-published.json` contains the full verified publisher configuration, including its published channels and scale/shift setting. It is not the old test dictionary with one renamed field. `if-first-synthetic-default-head.json` is explicitly synthetic: the original first DeepFloyd test dictionary with only the rejected `num_attention_heads` field removed, so the installed constructor's declared default is exercised. The two `original-invalid` dictionaries preserve the exact previously rejected inputs. Their actual capture and source-verification receipts are stored under `verification/receipts/S8-head-input-*` and `S8-publisher-head-input-independent`.
