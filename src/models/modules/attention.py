import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


        
class LabelAttentionHierarchical(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int,permutation_matrices):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)

        self.full_embedding = nn.Linear(projection_size, num_classes)

        self.embedding_fourth = nn.Linear(projection_size, 1041)
        self.embedding_third = nn.Linear(projection_size, 363)
        self.embedding_second = nn.Linear(projection_size, 77)
        self.embedding_one = nn.Linear(projection_size, 12)
        self.third_linear = nn.Linear(input_size, num_classes)
        self.permutation_matrix_1,self.permutation_matrix_2,self.permutation_matrix_3,self.permutation_matrix_4,self.permutation_matrix_all=permutation_matrices
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        # Combine the embeddings with permutation matrices here
        att_weights = self.full_embedding(weights)@ self.permutation_matrix_all
        att_weights += self.embedding_fourth(weights) @ self.permutation_matrix_4
        att_weights += self.embedding_third(weights) @self.permutation_matrix_3
        att_weights +=  self.embedding_second(weights)@self.permutation_matrix_2
        att_weights += self.embedding_one(weights) @ self.permutation_matrix_1
        
        
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x

   
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)
        torch.nn.init.normal_(self.full_embedding.weight, mean, std)
        torch.nn.init.normal_(self.embedding_fourth.weight, mean, std)
        torch.nn.init.normal_(self.embedding_third.weight, mean, std)
        torch.nn.init.normal_(self.embedding_second.weight, mean, std)
        torch.nn.init.normal_(self.embedding_one.weight, mean, std)


class CAMLAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.first_linear.weight)
        self.second_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.second_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CAML attention mechanism

        Args:
            x (torch.Tensor): [batch_size, input_size, seq_len]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = torch.tanh(x)
        weights = torch.softmax(self.first_linear.weight.matmul(x), dim=2)
        weighted_output = weights @ x.transpose(1, 2)
        return (
            self.second_linear.weight.mul(weighted_output)
            .sum(2)
            .add(self.second_linear.bias)
        )
