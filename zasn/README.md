# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles 
**[Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/pdf/2306.00989)** 
# Next-ViT 
**[Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501)**

## Cel
To repozytorium implementuje hybrydę dwóch wymienionych powyżej modeli.
Projekt został zrealizowany przez Dominika Słodkowskiego i Dominikę Milewską na potrzeby
projektu na przedmiot Zaawansowane Architektury Sieci Neuronowych.

## Sposób korzystania
W pierwszej kolejności trenujemy autoenkoder. Służy do tego skrypt train_mae.py. 
Skrypt powinien być uruchamiany z wiersza poleceń z obowiązkowym podaniem
ścieżki do katalogu z obrazami (w naszym wypadku podzbiór ImageNetu). W wyniku otrzymamy katalog z ostanim
najlepszym checkpointem oraz plik .txt z logami treningu.

Następnie prowadzimy finetuning modelu Hiera Tiny przy użyciu uzyskanego wcześniej checkpointu jako 
punktu startowego. Robimy to przy użyciu skryptu train_finetune.py, wywoływanego z wiersza poleceń
obowiązkowo z podaniem ścieżki do katalogu z obrazami i ścieżki do checkpointu. W wyniku otrzymamy katalog z ostanim
najlepszym checkpointem oraz plik .txt z logami treningu.

Kod ewaluacji znajduje się w pliku eval.py, który powinien być analogicznie wywoływany z wiersza poleceń.

Kod do generowania wykresów znajduje się w pliku plots.py.