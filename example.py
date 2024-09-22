from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "vinai/RecGPT-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda")

prompts = [
    "### Instruction:\nPredict the next item.\nGiven the interaction history of a user with food recipes as follows:\n\nTitle: white cheddar potato gratin\nTitle: possum s cream cheese chocolate chip bread\nTitle: zucchini lasagna lasagne low carb\nTitle: sweet potato and apple casserole\n\n### Response:",
    "### Instruction:\nPredict rating for the last item.\nGiven the interaction history of a user with books as follows:\n\nTitle: Rebooting My Brain: How a Freak Aneurysm Reframed My Life; Author: Visit Amazon's Maria Ross Page; Review: I would recommend this book to everyone! As I read the last page I felt that I wanted it to continue with anniversary updates. There is so much hope wry yen within the pages.; Rating: 5.0/5.0\nTitle: Summer of Firefly Memories (The Loon Lake Series); Author: Visit Amazon's Joan Gable Page; Review: Not as well written or edited as I would have liked but the story did what the author hoped it would do---gave you pause to reflect your own 'firefly memories'. this makes it a success. I look forward to the sequel,; Rating: 5.0/5.0\nTitle: Jon Stewart: Beyond The Moments Of Zen; Author: Visit Amazon's Bruce Watson Page; Review: Jon Stewart is a gift to the ages. Thank you ,Thank you,Thank you. My ears are never tired! long may he rage.; Rating: 5.0/5.0\nTitle: How to Succeed at Aging Without Really Dying; Author: Visit Amazon's Lyla Blake Ward Page; Review: Wonderful, funny, accurate!; Rating: 5.0/5.0\nTitle: Death of an Expert Witness (Adam Dalgliesh Mystery Series #6); Author: Visit Amazon's P. D. James Page; Review: You can always depend on a good read from PD James!; Rating:\n\n### Response:",
]

for prompt in prompts:
    input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.batch_decode(
        output_ids[:, input_ids.size(1):], skip_special_tokens=True
    )[0].strip()
    print(response)
