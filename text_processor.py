# %%
from transformers import BertTokenizer
from transformers import BertModel
# %%

class TextProcessor:
    """
    This class can be called on a string. It performs the necessary transformations so a model can be called on an text desciption. 
    """

    def __init__(self,
                 max_length: int = 50):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, text:str):
        encoded = self.tokenizer.batch_encode_plus(
            [text], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key: torch.LongTensor(value)
                   for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(
                **encoded).last_hidden_state.swapaxes(1, 2)
        return description
# %%
if __name__=="__main__":
    from neural_network_models import TextModel
    import pickle
    import torch

    device = torch.device('cpu')
    with open('models/decoder.pickle', 'rb') as f:
        decoder = pickle.load(f)


    model = TextModel(50, decoder)
    model.load_state_dict(torch.load(
        "models/nlp_model_state.pt", map_location=device))
    model.eval()

    text = TextProcessor()(input("Enter description: "))
    model.predict_classes(text)
# %%
