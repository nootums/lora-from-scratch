# Lora from scratch
Building LoRA adapters from scratch using pytorch and Llama-3.2-1B-instruct LLM model.

# Defining the LoraModule class
```
class LoraModule(nn.Module):
    def __init__(self, orig_module:nn.modules.linear.Linear, r:int=8, alpha:float=1.0):
        assert type(orig_module)==nn.modules.linear.Linear, "Original module should be a Linear layer"
        assert type(r)==int, "r(rank) should be an integer"
        assert type(alpha)==float, "alpha should be a float"
        super().__init__()
        self.alpha = alpha
        self.original_module = orig_module
        self.original_module.requires_grad = False
        orig_in_features = orig_module.in_features
        orig_out_features = orig_module.out_features
        self.lora_module = nn.Sequential(nn.Linear(orig_in_features, r, bias=False), nn.Linear(r, orig_out_features, bias=False))
        self.lora_module[0].weight = nn.Parameter(self.lora_module[0].weight)
        self.lora_module[1].weight = nn.Parameter(self.lora_module[1].weight)
        return
    
    def forward(self, x, *args, **kwargs):
        outs = self.original_module(x) + self.alpha*self.lora_module(x)
        return outs
    
    def set_alpha(self, new_alpha:float):
        assert type(new_alpha)==float, "New alpha value must be a float"
        self.alpha = new_alpha
        return
```
This class takes in a `nn.Linear` layer, and encapsulates it in a Lora class. The class is comprised of the original module(with `require_grad` set to `False`), and an nn.Sequential module, which is comprised of the LoRA matrices. Shape of the modules are `(Linear in features, r)` and `(r, Linear out features)`, where r is the rank of the adapter. The `forward` method claculates the projections based on the LoRA module, and adds it to the projection of the original Linear layer, scaled by a scalar called `alpha`. 

# Converting the model to a LoRA model
I go through each named parameter in the LLM model, and replace every `nn.Linear` module with the module defined above.

# Building the dataset
For the dataset, I am using the [English Phrases](https://huggingface.co/datasets/Abirate/english_quotes) dataset, which contains an english phrase, the Author, and the tags related to the phrase. While creating the text dataset, I took each sample and concatenated the "quote" and "tags" with the following separator: ` | tags: `. This separator will help us during training, and will also help us prompt the LLM to provide the tags while inferencing. 

Once I have prepared each sample with this format, I am then preprocessing the data by tokenizing the text(and appending the <eos> token) using the model's tokenizer, and storing each sample in a dictionary with the following elements:

- input_tokens : List of tokens *except* the <eos> token. This will be our input text.
- output_tokens : List of tokens *except* the <bos> token. This will be our target text.
- sep_token_index : Index of the '|' separator in the input sequence, this will help us to optimize the model finetuning later.


# Training
For the purposes of this project, I am using the [Llama-3.2-1b-instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model as it is small, and can be finetuned easily. We first take our model, convert it to a LoRA-fied model, and set the alpha to 0.125(this will be doubled in each epoch). If the alpha is low, the model outputs will not be gibberish, and the LoRA layers can be trained slowly. After each epoch I am doubling the alpha, as I hope the LoRA linear layers will start learning and adding meaningful adjustments to the projections, instead of adding random noise(on a side note, running the newly converted lora model on relatively low alphas can yield to interesting results. It's hilarious seeing the LLM have a "concussion").


While training, we will use the crossentropy loss on the sigmoid of the model outputs. This where the sep_token_index from the dataset comes into picture, as we will only calculate the loss on token predictions the come *after* the separator, as we want the lora adapters to learn to generate tags, so it doesn't make sense for us to penalize it for token predictions that are the part of the quote, This helps ensure that the new weights do not interfere with the quote part of the sentence, but only learn to generate the tags for the quote after the separator. Other than this small training quirk(and some minor training optimizations), the training pipeline is quite generic, where the hyperparameters, loss functions and optimizers can be changed as one sees fit.

# Results
This is able to learn the expected output format, which is strings in a list format, and we know the training is successful as the base model provides tags similar to instagram with hastags in them, but on loading the lora aapters, it is able to provide an ordered list, including the \<eos\> token.

Here are sample outputs:

```
Example 1:
Base model : <|begin_of_text|>Playing the bass guitar is fun | tags: bass, music, guitar, playing, fun\n\nI've always been fascinated by the bass guitar.
Lora model : <|begin_of_text|>Playing the bass guitar is fun | tags: bass guitar', 'play', 'rocking', 'sarcasm']<|eot_id|>

Example 2:
base model : <|begin_of_text|>Don't cry because its over, smile because it happened | tags: #relaxation, #selfcare, #mentalhealthmatters, #mindfulness, #
Lora model : <|begin_of_text|>Don't cry because its over, smile because it happened | tags:', 'humor', 'smile']<|eot_id|> 
```

In both these examples, the base model doesn't provide the tags in the correct format, and also does not stop generation by producing an \<eos\> token; while the lora finetuned model is able to do so.

Also, after dumping the lora weights, we can see that the additional weights only require 22.7MB of storage. I am uploading the lora weights in this repo in case any of you would like to load these weights and play around with them.


# Closing thoughts
This experiment was a success in my books, as even though there are some things that could be improved, this excercise proved that the implementation works, and provided with enough compute, along with high quality(and quantity) data, we would we able to create good quality LoRA adapters, even for complex tasks. Also, since the nature of generation was not instruction based, I feel we would've had better results with the non instruct Llama-3.2-1b model. 


Thanks for checking out this repo and I hope you learnt something from this :)