import torch


class Conversation(object):
    def __init__(self, model, tokenizer, model_max_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.conversation = []  # [[usr1,sys1], [usr2,sys2],...]
        self.device = device

    def _build_chat_history(self, **kwargs):
        chat_history = ''
        for j in range(len(self.conversation) - 1):
            i = self.conversation[j]
            chat_history += '[usr] '
            chat_history += i[0]
            chat_history += ' [sys] '
            chat_history += i[1]
            chat_history += ' '

        chat_history += '[usr] '
        chat_history += self.conversation[-1][0]

        chat_history += '[qst] '
        if 'qst' in kwargs:
            chat_history += kwargs['qst']
            chat_history += ' '
        else:
            chat_history += '[None]'
            chat_history += ' '

        chat_history += '[bhv] '
        if 'bhv' in kwargs:
            chat_history += kwargs['bhv']
        else:
            chat_history += '[None]'

        return chat_history


    def add_user_input(self, new_usr_input, **kwargs):
        self.conversation.append([new_usr_input, None])
        chat_history = self._build_chat_history(**kwargs)

        input_ids = self.tokenizer(chat_history).input_ids
        while len(input_ids) > self.model_max_length:
            if self.conversation.__len__() > 1:
                self.conversation.pop(0)
                chat_history = self._build_chat_history(**kwargs)
                input_ids = self.tokenizer(chat_history).input_ids
            else:
                input_ids = input_ids[-self.model_max_length:]


        input_ids = torch.tensor(input_ids).view(1, -1).to(self.device)
        out_ids = self.model.generate(input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(out_ids[0], skip_special_tokens=True)
        response = self.tokenizer.convert_tokens_to_string(tokens).strip()

        self.conversation[-1][1] = response
        if 'print_conv' in kwargs and kwargs['print_conv'] == True:
            print(f'''
            >> User: {new_usr_input}
            >> Bot: {response}
            ''',flush=True)
        return response

    def clear_dialog_history(self):
        self.conversation = []







