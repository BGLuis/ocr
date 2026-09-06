<div align="center">

<!-- Badges de Status do GitHub -->

![GitHub Stars](https://www.shieldcn.dev/github/stars/bgluis/ocr.svg?variant=secondary&size=sm)
![GitHub Forks](https://www.shieldcn.dev/github/forks/bgluis/ocr.svg?variant=secondary&size=sm)
![Watchers](https://www.shieldcn.dev/github/watchers/bgluis/ocr.svg?variant=secondary&size=sm)
![Contributors](https://www.shieldcn.dev/github/contributors/bgluis/ocr.svg?theme=emerald&size=sm)
![License](https://www.shieldcn.dev/github/license/bgluis/ocr.svg?variant=ghost&size=sm)

<br/>

<!-- Badges das Tecnologias Utilizadas -->

![Rust](https://www.shieldcn.dev/badge/Rust-1.95%2B-000000.svg?logo=rust&variant=branded&size=sm)
![ONNX Runtime](https://www.shieldcn.dev/badge/ONNX_Runtime-static-005CED.svg?logo=onnx&variant=branded&size=sm)
![Linux](https://www.shieldcn.dev/badge/Linux-Wayland-FCC624.svg?logo=linux&variant=branded&size=sm)

  <h3>ocr</h3>
  OCR multilíngue, local e offline: selecione uma região da tela e o texto vai para a área de transferência.
</div>

<p align="center">
  <b>Português (Brasil)</b> &nbsp;·&nbsp; <a href="README.en.md">English</a>
</p>

# 📖 Sobre

`ocr` é uma ferramenta de reconhecimento óptico de caracteres para uso diário no
desktop Linux/Wayland. Um **daemon residente** (`ocrd`) mantém os modelos
carregados na memória e um **cliente minúsculo** (`ocrd-client`) envia uma imagem
por um socket Unix e imprime o texto reconhecido. O caso de uso central é um
atalho de teclado: recorte uma parte da tela, o texto aparece na área de
transferência.

Por baixo:

- **Rust puro em runtime** — nenhum Python, nenhuma chamada de rede. É um
  _workspace_ Cargo com quatro _crates_: `ocr-proto` (tipos do protocolo),
  `ocr-core` (única fronteira que toca em `oar-ocr`/`ort`), `ocrd` (daemon) e
  `ocrd-client` (cliente).
- **Modelos PP-OCRv5** servidos via [`oar-ocr`](https://crates.io/crates/oar-ocr)
  sobre o [ONNX Runtime](https://onnxruntime.ai/), que é **linkado
  estaticamente** — os binários instalados rodam de qualquer lugar sem `.so`
  vizinho.
- **Um detector + três reconhecedores** residentes (~41 MiB): CJK (japonês,
  chinês, inglês), coreano e latino (português, espanhol, francês, alemão).
- **Idioma automático com _fallback_ por confiança** — em `--lang auto`, o
  reconhecedor CJK roda em todas as caixas; as caixas de baixa confiança
  (< 0.6) são repassadas aos reconhecedores coreano e latino, e o melhor
  resultado vence.
- **CPU por padrão.** Aceleração por GPU é opção explícita via
  `--accel auto|cpu|cuda|openvino|webgpu|directml`, sempre sondada antes de ser
  usada e fatal se indisponível (sem _fallback_ silencioso).
- **Integração com Wayland**: `grim` + `slurp` + `wl-clipboard`, _binds_ de
  Hyprland e uma _unit_ `systemd --user` prontas em `contrib/`.

O racional de projeto e as alternativas descartadas estão em
[`docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md`](docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md).

# 📋 Motivo

Criei este projeto para remover um bloqueio pessoal: não conseguir ter — nem
fazer — um OCR local que realmente prestasse. E, junto disso, para reconstruir
uma função de que eu gostava e usava bastante no PowerToys, agora no meu próprio
ambiente Linux/Wayland.

# 💻 Como iniciar

### Requisitos

- [Rust](https://www.rust-lang.org/tools/install) 1.95+ (edição 2021; a máquina
  de referência usa 1.97).
- Conexão à internet **na primeira compilação** (o `ort` baixa o ONNX Runtime) e
  no primeiro `ocrd --warm` (~41 MiB de modelos PP-OCRv5, gravados em `~/.oar`).
- Para o fluxo "print → OCR → área de transferência" no Wayland:
  [grim](https://sr.ht/~emersion/grim/),
  [slurp](https://github.com/emersion/slurp),
  [wl-clipboard](https://github.com/bugaevc/wl-clipboard).
- Opcional — GPU: NVIDIA com [cuDNN 9](https://developer.nvidia.com/cudnn) para
  `--features cuda`. Intel (`openvino`) e Windows (`directml`) exigem um ONNX
  Runtime de sistema — ver a tabela de aceleração no fim desta seção.

### Instalação

1. Clone o repositório do projeto:

```sh
git clone https://github.com/bgluis/ocr.git
```

2. Navegue até o diretório do projeto:

```sh
cd ocr
```

3. Escolha **um** dos métodos abaixo.

#### Método 1 — `cargo install` (recomendado)

Instala os dois binários em `~/.cargo/bin` e baixa os modelos uma vez:

```sh
cargo install --path crates/ocrd          # -> ~/.cargo/bin/ocrd        (~29 MiB)
cargo install --path crates/ocrd-client   # -> ~/.cargo/bin/ocrd-client (~1 MiB)
ocrd --warm                               # baixa os modelos para ~/.oar

# opcional: wrapper que sobe o ocrd sob demanda, sem systemd
install -Dm755 contrib/ocr ~/.local/bin/ocr
```

#### Método 2 — compilar a partir do código-fonte

```sh
cargo build --release                     # LTO, binários em ./target/release
cargo run -p ocrd -- --warm               # baixa os modelos
./target/release/ocrd --accel auto        # sobe o daemon
```

Para GPU, adicione a _feature_ correspondente ao construir o `ocrd`:
`cargo build --release -p ocrd --features cuda` (ou `openvino` / `directml`).

### Como usar

```sh
# 1. Deixe o daemon rodando (terminal ou systemd)
ocrd --accel auto

# 2. Em qualquer lugar: recorte a tela, reconheça, copie
grim -g "$(slurp)" - | ocrd-client --lang auto | wl-copy

# Ou a partir de um arquivo, com a resposta JSON completa
ocrd-client --file shot.png --lang latin --json
```

Instalar o serviço e os atalhos do Hyprland:

```sh
cp contrib/ocrd.service ~/.config/systemd/user/ && systemctl --user enable --now ocrd
cat contrib/hyprland.conf.example >> ~/.config/hypr/hyprland.conf
```

Opções principais do `ocrd-client`:

| Flag          | Significado                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `--lang`      | `auto` (padrão), `cjk` / `ja` / `zh` / `en`, `korean` / `ko`, `latin` / `pt` / `es` / `fr` / `de` |
| `--file PATH` | lê a imagem de um arquivo em vez do stdin                                                         |
| `--json`      | resposta completa (linhas, confiança por linha, _bboxes_, tempo)                                  |
| `--verbose`   | confiança por linha no stderr, texto no stdout                                                    |
| `--ping`      | sai com 0 se o `ocrd` está acessível; não lê imagem                                               |

### Configuração por ambiente

O projeto não usa `.env`; o comportamento é ajustado por estas variáveis:

| Variável                   | Padrão                       | Função                                                                                     |
| -------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------ |
| `OCRD_SOCKET`              | `$XDG_RUNTIME_DIR/ocrd.sock` | Caminho do socket Unix                                                                     |
| `OCRD_ACCEL`               | `auto`                       | `auto\|cpu\|cuda\|openvino\|webgpu\|directml`                                              |
| `OAR_HOME`                 | `~/.oar`                     | Cache dos modelos ONNX                                                                     |
| `OCRD_BIN` / `OCRD_CLIENT` | achados no `PATH`            | Binários usados pelo wrapper `contrib/ocr`                                                 |
| `OCRD_OPENVINO_DEVICE`     | `GPU`                        | Dispositivo alvo do OpenVINO                                                               |
| `RUST_LOG`                 | `ocrd=info,ocr_core=info`    | Verbosidade de log                                                                         |
| `OCR_REQUIRE_MODELS`       | —                            | Se definida, os testes de reconhecimento falham (em vez de pular) quando os modelos faltam |

### Testes

```sh
cargo test                                        # unitários + lógica, sem modelos

cargo run -p ocrd -- --warm                       # baixa os modelos
OCR_REQUIRE_MODELS=1 cargo test -p ocr-core --test recognition
```

### Aceleração por GPU

| Fornecedor                   | Flag               | Build                 | Situação com o ONNX Runtime pré-compilado                                                                |
| ---------------------------- | ------------------ | --------------------- | -------------------------------------------------------------------------------------------------------- |
| **NVIDIA**                   | `--accel cuda`     | `--features cuda`     | Funciona, dado cuDNN 9 + `libcudart.so.12` no host.                                                      |
| **Intel** (iGPU / Arc / NPU) | `--accel openvino` | `--features openvino` | Não linka com o ORT baixado; precisa de um ONNX Runtime de sistema com OpenVINO (`ORT_STRATEGY=system`). |
| **AMD** no Linux             | `--accel webgpu`   | _(sem flag)_          | Precisa de um ONNX Runtime compilado do zero com WebGPU/Dawn; até lá, erro limpo e fatal.                |
| **Windows**, qualquer GPU    | `--accel directml` | `--features directml` | Como o OpenVINO: exige um ORT de sistema com o EP DirectML. Só Windows.                                  |

`--accel auto` só tenta CUDA e cai para CPU (com o motivo registrado no log);
nunca escolhe um _backend_ de fornecedor silenciosamente.

# 🤝 Contribuidores

 <a href="https://github.com/bgluis/ocr/graphs/contributors">
   <img src="https://contrib.rocks/image?repo=bgluis/ocr"/>
 </a>
