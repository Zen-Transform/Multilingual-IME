

class KeystrokeTokenizer():
    """
    A tokenizer for keystrokes, converting strings to token IDs for use in machine learning models.
    This tokenizer is designed to handle various keystrokes, including letters, numbers,
    punctuation, and special characters.

    """
    key_labels = [
        "PAD", "<SOS>", "<EOS>", "<UNK>",
        "`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=",
        "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", "\\",
        "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", 
        "z", "x", "c", "v", "b", "n", "m", ",", ".", "/",
        "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+",
        "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "{", "}", "|",
        "A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "\"",
        "Z", "X", "C", "V", "B", "N", "M", "<", ">", "?",
        "®", " "
    ]

    @classmethod
    def tokenize(cls, input_keystrokes: str) -> list[str]:
        """
        Tokenize the input string into a list of tokens

        Args:
            input_keystrokes (str): The input string

        Returns:
            list: The list of tokens
        """

        token_list = []
        token_list.append("<SOS>")

        for key in input_keystrokes:
            if key not in cls.key_labels:
                token_list.append("<UNK>")
            else:
                token_list.append(key)

        token_list.append("<EOS>")
        return token_list

    @classmethod
    def token_to_ids(cls, token_list:list[str]) -> list[int]:
        """
        Convert a list of tokens to a list of token ids

        Args:
            token_list (list[str]): The list of tokens

        Returns:
            list[int]: The list of token ids
        """

        id_list = []
        for token in token_list:
            assert token in cls.key_labels, f"Error: can not convert token '{token}' is not on list"
            id_list.append(cls.key_labels.index(token))
        return id_list


    @classmethod
    def key_labels_length(cls):
        """
        Returns the length of the key labels list.
        This is used to determine the size of the one-hot encoding vector.
        """
        return len(cls.key_labels)

if __name__ == '__main__':
    input_str = "><z;6ru.4y9 － u3s061j"
    tokens_list = KeystrokeTokenizer.tokenize(input_str)
    ids_list = KeystrokeTokenizer.token_to_ids(tokens_list)

    # embedding = torch.eye(KeystrokeTokenizer.key_labels_length())[ids_list]
    print(input_str)
    print(tokens_list)
    print(ids_list)
    # print(embedding)
    # print(embedding.shape)
