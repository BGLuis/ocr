# OCR multilíngue de uso diário — viabilidade e escolha da linguagem de implementação

| Campo | Valor |
|-------|-------|
| **Status** | ❌ Não iniciado — o diretório do projeto está vazio, não há uma linha de código |
| **Cobertura** | ~0 % (0 de 8 fases) |
| **Esforço [modelado]** | 2–3,5 dias-dev no mínimo viável (imagem, atalho global, latim + japonês + chinês); 8–12 d no escopo sem GUI; 11–17 d com GUI |
| **Depende de** | Nada. Toda a toolchain necessária já está instalada na máquina |
| **Atenção** | O pedido de "não priorizar Python" é **tecnicamente atendível sem perda de qualidade** — a seção 2.1 explica por quê |
| **Atenção** | A GPU **não é requisito**. O padrão é CPU; a RTX 3060 entra por detecção ou por `--accel` — seção 2.9. Referência de experiência: PowerToys Text Extractor — seção 2.8 |

---

## 1. Estado atual — evidências

### 1.1 O repositório está vazio

Não existe código a auditar. A prova:

```bash
find /home/luis/Documents/hand-on/OCR -maxdepth 3 -not -path '*/.git/*'
# → /home/luis/Documents/hand-on/OCR   (apenas o próprio diretório)
```

Nenhum `Cargo.toml`, nenhum `package.json`, nenhum `pyproject.toml`, nenhum `.git`.
Toda a análise abaixo é sobre **o que a máquina oferece** e **o que o ecossistema
publica**, não sobre código existente. Este relatório não cita `arquivo:linha` porque
não há arquivo — cita comandos com a saída real e URLs verificáveis.

### 1.2 O que a máquina já tem — e é bastante

Verificado por execução direta, em 06/09/2026:

| Item | Versão medida | Relevância |
|---|---|---|
| `rustc` / `cargo` | 1.97.1 | Candidato principal (seção 2.2) |
| `go` | 1.27.0 | Candidato descartado |
| `node` | 26.8.1 · `deno` 2.9.6 | Candidato descartado |
| `python3` | 3.14.7 · `uv` 0.12.9 | ⚠️ Ver a armadilha em 1.3 |
| `gcc` 16.2.1 · `clang` 22.1.8 · `cmake` 4.4.3 | — | Suficiente para qualquer crate `-sys` |
| `ffmpeg` / `ffprobe` | n9.0.1, com hwaccel `cuda`, `vaapi`, `vulkan` | Pipeline de vídeo (seção 2.6) |
| `magick` (ImageMagick) | presente | Pré-processamento e testes |
| `grim` · `slurp` · `wl-copy` | presentes | Captura de região no Wayland (seção 2.7) |
| Sessão gráfica | `XDG_SESSION_TYPE=wayland`, `XDG_CURRENT_DESKTOP=Hyprland` | Define **como** o atalho global funciona |
| CPU | Intel i5-12600K, 16 threads, AVX2 | Inferência em CPU é viável |
| RAM | 62 GiB | Nenhuma restrição |
| GPU | NVIDIA RTX 3060, 12 GiB, driver 610.57.04 | Abre a porta para VLM local |
| CUDA Toolkit | 13.3 (`nvcc` V13.3.73) | Compatível com ONNX Runtime ≥ 1.27 |
| Fontes CJK | 80 famílias Noto CJK / Source Han | Permite gerar corpus sintético de teste |
| Disco livre em `/home` | 115 GiB de 913 GiB (88 % usado) | ⚠️ Apertado para modelos VLM |

### 1.3 ⚠️ O que a máquina não tem — três lacunas com consequência

**Primeira: o Tesseract instalado não sabe ler nenhuma das línguas pedidas.**

```bash
tesseract --list-langs
# → List of available languages in "/usr/share/tessdata/" (3):
#   eng
#   osd
#   por

tesseract ja_h.png stdout -l jpn
# → Error opening data file /usr/share/tessdata/jpn.traineddata
#   Failed loading language 'jpn'
```

Os pacotes existem nos repositórios oficiais (`tesseract-data-jpn`, `-jpn_vert`,
`-chi_sim`, `-kor`, todos na versão `2:4.1.0-5`), mas **não estão instalados** — e a
seção 2.3 argumenta que instalá-los é o caminho errado.

**Segunda: `cudnn` não está instalado.** O ONNX Runtime exige cuDNN 9.x para o
*execution provider* CUDA, e desde a versão 1.27 o pacote padrão é construído contra
CUDA 13.0 + cuDNN 9.x — que casa exatamente com o CUDA 13.3 da máquina. O pacote está
disponível em `extra/cudnn 9.25.1.1-1`; é **um `pacman -S` de distância**, não um
bloqueio. Sem ele, a inferência cai silenciosamente para CPU.

**Terceira — e esta é a armadilha real do caminho Python:** o Python do sistema é
3.14.7, e `paddlepaddle` **não publica wheel para cp314**. Medido consultando a API
do PyPI:

```
onnxruntime    v1.29.0   tags=[cp311, cp312, cp313, cp314]   cp314=SIM
torch          v2.14.0   tags=[cp310..cp314]                 cp314=SIM
paddlepaddle   v3.3.1    tags=[cp39, cp310, cp311, cp312, cp313]   cp314=NÃO
paddleocr      v3.7.0    py3/sdist                           cp314=NÃO
rapidocr       v3.9.2    py3                                 cp314=NÃO
```

Além disso, `rapidocr-onnxruntime` 1.4.4 declara `requires_python = <3.13,>=3.6`.
Ou seja: **um `pip install paddleocr` neste computador falha hoje**. É contornável
com `uv python pin 3.12`, mas é sintomático — o caminho Python já começa com uma
divergência de versão para administrar, e vai continuar tendo.

### 1.4 Medição: o que o Tesseract atual devolve para CJK

Gerei quatro imagens sintéticas com `magick` e a fonte `NotoSansCJK-Regular.ttc`, e
rodei o `tesseract` 5.5.3 com o único modelo latino disponível. **Números medidos**,
não estimados:

| Imagem | Texto de entrada | Saída do Tesseract `-l eng` | Tempo |
|---|---|---|---|
| `en.png` | `The quick brown fox 2026` | `‘The quick brown fox 2026` | 0,156 s |
| `ja_h.png` | `吾輩は猫である。名前はまだ無い。` | `EEIIETH D, BEITEIEEL,` | 0,122 s |
| `zh.png` | `人工智能技术发展` | `ANIEERALE` | 0,096 s |
| `ko.png` | `안녕하세요 세계` | `CHASIMR MIA` | 0,103 s |

Duas leituras. A primeira é óbvia: para CJK sem o modelo certo, a saída é **ruído
puro** — nem um caractere correto. A segunda é mais interessante: mesmo em inglês
impresso perfeito, em fonte limpa, sem ruído, o Tesseract inseriu uma aspa curva
espúria no começo da linha. Esse é o piso de qualidade do motor, não o teto.

O tempo — ~0,1 s por linha isolada — é o **orçamento de latência a bater**: qualquer
solução mais lenta que isso vai parecer pior no uso diário, mesmo sendo mais correta.

---

## 2. As dez decisões de design

### 2.1 A linguagem do aplicativo não decide a qualidade do OCR — o modelo decide

A qualidade de um OCR moderno vem de dois artefatos: um modelo de **detecção** (acha
as caixas de texto) e um modelo de **reconhecimento** (lê cada caixa). Ambos são
arquivos `.onnx` — grafos de operadores, sem código Python dentro. O ONNX Runtime é
uma biblioteca C++ com API C estável, e **toda** linguagem com FFI executa esses
arquivos com exatamente a mesma aritmética.

O que o Python monopoliza é o **laboratório**: treinar, fazer *fine-tuning* e exportar
modelos (`paddle2onnx`, `optimum`). Nada disso acontece em tempo de execução — e aqui
nem a exportação é necessária: os modelos PP-OCRv5 e PP-OCRv6 **já estão publicados em
ONNX**, com tamanho conhecido, nos *releases* do `oar-ocr` e no Hugging Face.

A preferência por não usar Python **não custa acurácia**. Custa o pré e o
pós-processamento (redimensionamento, normalização, decodificação CTC, agrupamento de
caixas) na linguagem escolhida — e já existe crate que fez isso.

### 2.2 A linguagem: Rust

| Opção | O que é | Custo | Veredito |
|---|---|---|---|
| **Rust + `ort` + `oar-ocr`** | Binário único, ONNX Runtime via FFI, pipeline PP-OCR pronta | `ort` está em `2.0.0-rc.13` — nunca teve *release* estável | **Recomendada** |
| Python + `rapidocr` / `paddleocr` | Caminho mais curto até o primeiro resultado | Sem wheel cp314 (1.3); *startup* do interpretador; distribuição ruim | Rejeitada como *runtime* — **mantida como laboratório** |
| Go + `onnxruntime_go` | FFI estável, binário único, *cross-compile* fácil | Nenhuma biblioteca de pipeline OCR madura; escreveria detecção DB e decodificação CTC do zero | Rejeitada: 3–5 d [modelado] de trabalho puro de reimplementação |
| Node/TS + `onnxruntime-node` | 6,81 M *downloads*/semana; `tesseract.js` 2,88 M/semana | Idem Go quanto ao pipeline; e `tesseract.js` herda o motor medido em 1.4 | Rejeitada |
| C# / Java + RapidOCR | RapidOCR mantém *ports* oficiais nessas duas linguagens | *Runtime* extra no Linux; integração fraca com Hyprland | Rejeitada |
| C++ + RapidOCR C++ | Implementação de referência do RapidOCR | Maior custo de desenvolvimento sem ganho de desempenho relevante | Rejeitada |

O argumento decisivo a favor de Rust não é desempenho de inferência — o ONNX Runtime
é o mesmo binário C++ em todos os casos. É **maturidade do ecossistema para este
problema específico**, e ela é mensurável:

```
crate            versão      total downloads   últimos 90 d   atualizado
ort              2.0.0-rc.13      17.578.332      6.678.320    2026-07-28
oar-ocr          0.9.2               280.499         31.697    2026-08-18
rten             0.26.0            1.404.078        768.204    2026-08-29
ocrs             0.13.0              272.960        152.096    2026-08-29
ocrs-cjk         0.1.0                   328            328    2026-06-26
paddle-ocr-rs    0.6.1                18.078          4.125    2025-08-31
leptess          0.14.0              271.091        108.710    2023-02-21   (abandonada)
```

E há um precedente forte: **koharu** (`koharu-rs/koharu`, 5.495 estrelas,
MIT OR Apache-2.0) é um tradutor de mangá *local-first* escrito inteiramente em Rust
com Tauri, que faz detecção de balões, OCR japonês, *inpainting* e tradução por LLM.
O `Cargo.toml` dele mostra a arquitetura real: `koharu-torch-sys`, `koharu-llama-sys`
e um crate `koharu-runtime` que descreve a si mesmo como *"runtime package discovery,
downloading, and dynamic library loading"* com as *keywords* `cuda, libtorch`.

Isso ensina duas coisas de uma vez, e a segunda é desconfortável: **dá para fazer um
app de OCR japonês sério sem Python nenhum** — e o projeto mais bem-sucedido do gênero
**não ficou em Rust puro**; ele carrega libtorch e llama.cpp dinamicamente em tempo de
execução. Rust aqui é a linguagem de *orquestração*, não de inferência. Aceite isso
desde o início e o projeto anda; recuse e você vai reimplementar GEMM.

### 2.3 O motor: PP-OCRv5/v6 via `oar-ocr`, não Tesseract

| Opção | O que é | Custo | Veredito |
|---|---|---|---|
| **`oar-ocr` (Rust, ONNX Runtime)** | Pipeline PP-OCRv4/v5/v6 completa: detecção, reconhecimento, orientação de página e de linha, *unwarping*, *download* automático de modelos, *feature* `cuda` | 163 estrelas, Apache-2.0, último *push* 2026-08-20 — projeto jovem | **Recomendada** |
| Tesseract via `leptess` | Já instalado; `-data-jpn` a um `pacman` de distância | Qualidade medida em 1.4; `jpn_vert` tem falha documentada de orientação; `leptess` sem *release* desde 2023-02-21 | Rejeitada |
| `ocrs-cjk` (Rust puro, `rten`) | Zero dependências C/C++, roda em WASM, PP-OCRv5 ONNX | 328 *downloads* totais, 3 estrelas, *"early preview"*; `rten` é **CPU-only** | Rejeitada agora — reavaliar em 6 meses |
| RapidOCR (Python/C++/Java/C#) | 7.722 estrelas, Apache-2.0, modelos ONNX oficiais | Não tem *port* Rust oficial | Rejeitada como dependência, **aceita como fonte de modelos** |
| VLM local (PaddleOCR-VL 0,9 B, dots.ocr 3 B) | Lê a página inteira e emite Markdown estruturado | 12 GiB de VRAM comportam; latência de segundos, não de décimos | Rejeitada para o atalho diário — **considerar como modo "documento"** |
| API de nuvem (Google Vision, Azure) | Melhor acurácia absoluta | Manda **toda captura de tela** para terceiros; custo recorrente; inútil offline | Rejeitada |

Sobre a última linha, sendo explícito: a ferramenta proposta lê a tela do usuário
várias vezes por dia. Enviar isso para uma API é um vazamento contínuo de contexto
pessoal e profissional. Local é requisito, não preferência.

### 2.4 Multilíngue: um detector, três reconhecedores — todos residentes

A detecção é **agnóstica de idioma**: `pp-ocrv5_mobile_det.onnx` tem 4,6 MiB e acha
caixas de texto em qualquer escrita. Só o reconhecedor é específico. Tamanhos reais,
lidos da tabela de modelos do `oar-ocr`:

| Modelo | Tamanho | Cobre |
|---|---|---|
| `pp-ocrv5_mobile_det.onnx` | 4,6 MiB | Detecção — **compartilhado por todos** |
| `pp-ocrv5_mobile_rec.onnx` | 15,8 MiB | Chinês simplificado, tradicional, inglês e **japonês**, num modelo só |
| `korean_pp-ocrv5_mobile_rec.onnx` | 12,8 MiB | Coreano |
| `latin_pp-ocrv5_mobile_rec.onnx` | 7,7 MiB | Escrita latina com diacríticos — português, espanhol, francês, alemão |

Soma: **40,9 MiB** de modelos. Isso muda o desenho. Não existe "problema de seleção de
idioma": os quatro modelos cabem na RAM simultaneamente com folga absurda (62 GiB
disponíveis). O daemon carrega os quatro na inicialização, roda a detecção **uma vez**,
e então:

1. Roda `pp-ocrv5_mobile_rec` em todas as caixas — cobre o caso dominante (japonês,
   chinês, inglês) numa passada.
2. Para caixas cuja confiança média fique abaixo de um limiar, reprocessa com
   `korean` e `latin`, e fica com a maior confiança.

Esse *fallback* por confiança custa uma segunda passada apenas nas caixas duvidosas.
Também deve existir um atalho que **força** um idioma, porque o usuário quase sempre
sabe o que está olhando e forçar é mais rápido e mais correto que adivinhar.

⚠️ Uma advertência sobre a documentação oficial: a *model card* de
`latin_PP-OCRv5_mobile_rec` no Hugging Face diz, no texto em inglês, que o modelo
*"aims to support the recognition of Korean"* — enquanto a tabela de acurácia logo
abaixo, em chinês, mede `拉丁字母语言` (línguas de alfabeto latino), com 84,7 % de
acurácia por linha. É erro de copiar-e-colar do *card*, não do modelo. **Não confie
nas descrições em prosa dessas páginas; confie no dicionário de caracteres
(`dict.txt`) que acompanha cada modelo** — ele é a especificação real do que o
reconhecedor consegue emitir.

### 2.5 Japonês vertical e mangá: um caso à parte, e o mais caro

O PP-OCRv5 declara suporte a texto vertical, e a *model card* de `PP-OCRv5_mobile_rec`
publica 0,7577 de acurácia no cenário japonês (contra 0,8605 em chinês impresso).
Para menus, sites, legendas e documentos, é suficiente.

Para **mangá**, não é. Mangá tem furigana, texto sobre a arte, fontes manuscritas e
balões multilinha. O especialista do domínio é `kha-white/manga-ocr` (2.772 estrelas,
Apache-2.0): um ViT como *encoder* e um BERT japonês em nível de caractere como
*decoder*, que reconhece **um balão inteiro numa única passada**, sem quebrar em
linhas — exatamente o que um pipeline detecção-mais-linha não faz.

Existe exportação ONNX pronta em `onnx-community/manga-ocr-base-ONNX`, com os tamanhos
reais:

| Arquivo | Tamanho |
|---|---|
| `encoder_model.onnx` | 343 MB |
| `encoder_model_q4f16.onnx` | 49,8 MB |
| `decoder_model.onnx` | 117 MB |
| `decoder_model_q4f16.onnx` | 24,1 MB |

O custo escondido está na arquitetura: um `VisionEncoderDecoder` é **autorregressivo**.
Chamar `session.run()` uma vez não devolve texto — é preciso escrever o laço de
decodificação gulosa em Rust, alimentando o *decoder* com os tokens já emitidos, com
gestão de *KV cache*, até o token de fim. Isso é 2–3 dias-dev [modelado] de trabalho
que o pipeline PP-OCR não exige. **É o item mais caro do plano por unidade de valor —
e o único que se aplica a um nicho.** Deixe-o para o fim (F5) e só faça se o uso real
provar que o PP-OCRv5 não resolve o seu mangá.

### 2.6 Vídeo: o gargalo não é o OCR, é a deduplicação temporal

Um vídeo de 1080p a 24 fps tem 1.440 quadros por minuto. Rodar OCR em todos é
absurdo — e produz mil cópias da mesma legenda.

O pipeline correto tem quatro estágios, e três deles não são OCR:

1. **Amostragem** — `ffmpeg` extrai quadros a 2–4 fps da faixa inferior da imagem
   (`crop`), onde a legenda vive. Já há `hwaccel cuda` disponível no build instalado.
2. **Deduplicação por similaridade** — descarta o quadro se ele for quase idêntico ao
   anterior. É a técnica que o `VideOCR` usa (limiar de SSIM), e é o que derruba o
   volume em uma ou duas ordens de grandeza.
3. **OCR** — o mesmo núcleo da seção 2.4, sem nenhuma alteração.
4. **Agregação temporal** — agrupa leituras consecutivas iguais numa única legenda com
   `início → fim`, resolve divergências por voto majoritário entre quadros do mesmo
   grupo (isto **melhora** a acurácia em relação a um quadro único) e emite `.srt`.

O estágio 4 é o que separa uma pilha de texto solto de uma legenda usável, e é
inteiramente lógica de programa — nenhum modelo envolvido. É onde Rust rende mais.

### 2.7 Processo residente com atalho global, não CLI fria

Para uso diário, o custo dominante **não é a inferência, é a inicialização**: carregar
o ONNX Runtime e quatro modelos a cada acionamento torna a ferramenta insuportável.
A arquitetura é um daemon com os modelos quentes, mais um cliente minúsculo:

```
bind = SUPER SHIFT, T, exec, grim -g "$(slurp)" - | ocrd-client --lang auto | wl-copy
```

Todas as três ferramentas dessa linha — `grim`, `slurp`, `wl-copy` — **já estão
instaladas** (verificado em 1.2), e a sessão é Hyprland/Wayland, onde `hyprland.conf`
resolve o atalho global sem nenhuma biblioteca. O cliente fala com o daemon por
*socket* Unix. Nenhuma GUI é necessária para o produto ser útil no dia 1 — e é por isso
que a GUI é a fase F7, opcional, e não a F1.

### 2.8 Referência de UX: PowerToys Text Extractor — e o que ele não prova

O `Text Extractor` do Microsoft PowerToys é o alvo de experiência a copiar: um atalho
global abre uma sobreposição, o usuário arrasta um retângulo, o texto cai no
*clipboard*. É exatamente o fluxo da seção 2.7. Vale entender por que ele parece não
ter fricção nenhuma — e por que isso **não** se traduz num motor superior.

O PowerToys **não tem motor de OCR próprio**. Ele é uma casca fina — seleção de
região mais cópia — sobre `Windows.Media.Ocr`, a API de OCR embutida no Windows desde
2016. Segundo a documentação da Microsoft, esse motor *"runs entirely on the device
without requiring an Internet connection"*: **é local e não usa GPU**. É um pipeline
clássico e enxuto (CNN + segmentação + modelo de linguagem por idioma), não um
transformer — por isso é minúsculo e rápido em CPU. Os idiomas entram como *"OCR
language packs"* de poucos MB via `Add-WindowsCapability`; não há *download* de
`.onnx`, não há escolha de *runtime*. São ~25 idiomas, incluindo `ja-JP`, `ko-KR`,
`zh-Hans`, `zh-Hant`.

A ausência de fricção vem de **empacotamento do SO**, não de mágica no modelo: a
Microsoft colou o motor e o sistema de *language packs* dentro do Windows, e o autor
do app só chama `OcrEngine.TryCreateFromLanguage()`. E o motor tem limites duros que
raramente incomodam porque texto de tela é o caso mais fácil de OCR que existe:
afinado para texto renderizado, nítido e horizontal; degrada em foto, rotação e
manuscrito; e nos casos caros deste relatório — japonês vertical (2.5) e legenda sobre
vídeo movimentado (2.6) — ele sofreria tanto quanto qualquer outro. A própria
documentação avisa: *"the produced text may not be perfect, so you have to do a quick
proof read"*.

A lição para este projeto tem dois lados:

| Parte do PowerToys | Traduz para Linux? |
|---|---|
| O fluxo atalho → *overlay* → região → *clipboard* | **Sim, e já existe pronto.** O `NormCap` (`dynobo/normcap`) faz isso no Wayland/Hyprland hoje, com Tesseract e `wl-clipboard`. Use-o como protótipo de referência da F2 antes de escrever a sobreposição própria |
| O motor pequeno, só CPU, sem gerência de modelo | **Sim, mas o app carrega o peso.** O análogo direto de `Windows.Media.Ocr` é o PP-OCRv5 *mobile* em CPU (2.3, 2.4): mesmo ponto de projeto, mesma dispensa de GPU. A diferença é que no Windows o SO entrega os *packs*; aqui `ocr-core` embarca os ~40,9 MB e faz o próprio *download*/cache — trabalho de empacotamento, não de acurácia |

Ou seja: o `Text Extractor` **confirma** o mínimo viável deste relatório em vez de
apontar um caminho melhor. O que ele não cobre — vídeo, vertical, mangá — é
precisamente onde o motor do Windows também pararia.

### 2.9 GPU quando o stack completo estiver presente, CPU por padrão, `--accel` para forçar

A máquina tem uma RTX 3060 de 12 GiB e CUDA 13.3 (1.2), mas **o padrão é CPU**, e essa
escolha é deliberada — não uma limitação.

**Por que CPU é o padrão e a linha de base.** Os modelos *mobile* têm 7–16 MB (2.4) e
o i5-12600K tem AVX2 e 16 *threads*; o Tesseract já processa uma linha isolada em
~0,1 s nesta máquina (1.4), e o PP-OCRv5 *mobile* está na mesma ordem. Para uma
captura de tela avulsa — o caso dominante — o custo é dominado pela inicialização, não
pela inferência (2.7), e a GPU quase não ajuda. É o mesmo ponto de projeto do
`Windows.Media.Ocr` (2.8): sem GPU, sem configuração, funciona em qualquer máquina.

**Por que a GPU é ativada por detecção, não ligada por padrão.** O *execution
provider* CUDA do ONNX Runtime exige `cudnn` 9.x e um *build* de ORT casado com a
versão do CUDA (1.3, seção 4). Se qualquer peça faltar, o ORT **cai para CPU em
silêncio** (seção 4). Então o daemon sonda o acelerador na inicialização e só usa
CUDA se o *provider* **realmente inicializou**, registrando no *log* qual *provider*
venceu — nunca presumindo.

**O controle:** uma *flag* `--accel {auto|cpu|cuda}` no daemon e no cliente.

| Valor | Comportamento |
|---|---|
| `auto` (padrão) | Tenta CUDA; se o *provider* não inicializar, cai para CPU e registra o motivo no *log* |
| `cuda` | **Exige** CUDA; se não inicializar, o processo falha alto (é a mitigação já listada na seção 4 para o *fallback* silencioso). Para bancada e para garantir que a GPU está de fato em uso |
| `cpu` | Força CPU. Para *benchmark* reprodutível, e para não disputar os 12 GiB de VRAM quando a GPU está ocupada com jogo, edição ou treino |

**Onde a GPU compensa de verdade:** nos fluxos em lote, não no atalho. Vídeo (2.6),
PDF de muitas páginas, e o laço autorregressivo do `manga-ocr` (2.5) — que faz dezenas
de passadas de *decoder* por região — são onde a GPU rende 3–10× [modelado] e paga a
complexidade. O `oar-ocr` já expõe isso pela *feature* `cuda` (2.3); o trabalho é a
sondagem e a *flag*, não a integração.

**Recomendação:** entregue com `auto`. Na F0, meça o ganho real CPU × CUDA nos modelos
*mobile* de fato usados. Se o ganho no caminho interativo for menor que ~2×, faça
`auto` reservar a GPU **apenas** para os fluxos em lote e manter o atalho sempre em
CPU — menos código quente, menos superfície de bug, e a ferramenta continua idêntica
numa máquina sem GPU.

### 2.10 O que não fazer

- **Não instale `tesseract-data-jpn` achando que resolve.** Resolve mal: a medição
  de 1.4 mostra o piso de qualidade do motor, e o `jpn_vert` tem falha conhecida de
  orientação — carregado como idioma secundário, o Tesseract lê o texto vertical na
  horizontal e falha. Em comparações publicadas de texto japonês vertical, o
  Tesseract fica em CER acima de 0,7 enquanto alternativas especializadas ficam na
  casa de 0,01. Não vale o dia de trabalho.
- **Não escreva o motor de OCR do zero.** Detecção DB, decodificação CTC e agrupamento
  de caixas são problemas resolvidos e chatos. Use `oar-ocr`; se ele se provar
  inadequado, aí sim descasque uma camada.
- **Não comece pela GUI.** O produto é um atalho de teclado.
- **Não faça vídeo antes de imagem funcionar.** Vídeo é imagem mais deduplicação; se o
  núcleo estiver instável, você vai depurar os dois ao mesmo tempo.
- **Não misture o laboratório com o runtime.** Se precisar de Python para exportar um
  modelo, faça num `uv` isolado com `python pin 3.12`, gere o `.onnx`, e **não deixe
  isso virar dependência do aplicativo**.

---

## 3. Plano de implementação

| Fase | Conteúdo | Esforço [modelado] |
|---|---|---|
| **F0** | *Spike*: `cargo add oar-ocr --features cuda,auto-download`; `pacman -S cudnn`; rodar o exemplo nas 4 imagens de teste de 1.4 e comparar com a tabela medida; medir o ganho CPU × CUDA nos modelos *mobile* (2.9) | 0,5 d |
| **F1** | Crate `ocr-core`: API `recognize(image, lang) -> Vec<TextLine>`; carregamento dos 4 modelos; *fallback* por confiança (2.4); sondagem de acelerador e *flag* `--accel {auto\|cpu\|cuda}` (2.9) | 1,5–2 d |
| **F2** | Binário CLI `ocrd-client`: stdin/arquivo → stdout; integração `grim`/`slurp`/`wl-copy`; *bind* no `hyprland.conf` | 0,5–1 d |
| **F3** | Daemon `ocrd`: *socket* Unix, modelos residentes, *systemd --user* | 1–1,5 d |
| **F4** | Detecção automática de escrita por confiança + atalhos por idioma forçado | 1 d |
| **F5** | `manga-ocr` em ONNX: laço de decodificação autorregressiva em Rust (2.5) | 2–3 d |
| **F6** | Pipeline de vídeo: `ffmpeg` → dedup SSIM → OCR → agregação temporal → `.srt` | 2–3 d |
| **F7** | GUI opcional (Tauri, como no koharu, ou `egui`): sobreposição do resultado, seleção, histórico | 3–5 d |

**Mínimo viável** (só o que foi pedido literalmente para imagens): F0 + F1 + F2
≈ 2,5–3,5 d.
**Escopo sem GUI:** F0–F6 ≈ 8,5–12 d.
**Escopo completo:** F0–F7 ≈ 11,5–17 d.

A ordem importa em dois pontos. F3 antes de F4 porque a troca de modelo por idioma só
faz sentido com os modelos já residentes — fazer o inverso obriga a reescrever o
carregamento. E F6 depois de F1 porque o vídeo consome a mesma API de `ocr-core`: se
ela mudar, o pipeline de vídeo é refeito.

---

## 4. Armadilhas

| Armadilha | Mitigação |
|---|---|
| `ort` está em `2.0.0-rc.13` e nunca teve *release* estável; a API muda entre *rc* | Fixe a versão exata no `Cargo.toml` (`=2.0.0-rc.13`) e isole todo uso de `ort` dentro de `ocr-core`. `oar-ocr` já faz essa camada por você |
| ONNX Runtime cai para CPU **em silêncio** quando o EP CUDA não carrega | Registre o *execution provider* efetivo no *log* de inicialização do daemon e falhe alto se `cuda` foi pedido e não foi obtido |
| CUDA 13.3 na máquina, mas ORT < 1.27 é construído para CUDA 12.x — misturar dá erro obscuro de símbolo | Confirme que a versão de ORT baixada pela *feature* `download-binaries` é ≥ 1.27; instale `extra/cudnn` (9.25.1.1) antes do primeiro teste |
| A descrição em prosa das *model cards* do PP-OCRv5 tem erro de copiar-e-colar (2.4) | Valide cada modelo pelo `dict.txt` que o acompanha, não pela descrição |
| Modelo errado devolve texto plausível e errado, sem erro | Exponha a confiança por linha na saída (`--json`); trate `< 0,6` como suspeito |
| Captura no Wayland não funciona com ferramentas X11 | `grim`/`slurp` já são as corretas e já estão instaladas; `xdotool` presente é uma pista falsa, é X11 |
| `magick` sozinho não normaliza a imagem para o modelo | O pré-processamento (escala, *padding*, normalização) é parte do contrato do modelo; deixe `oar-ocr` fazer, não invente |
| Texto branco com borda escura sobre vídeo claro derruba a detecção | Amostre o quadro em duas versões (original e invertida) e fique com a de maior confiança — barato, e resolve a maioria dos casos |
| Disco em 88 % — modelos VLM de 1–3 B ocupam GiB | Se for testar VLM (2.3), aponte o cache do Hugging Face para fora de `/home` ou libere espaço antes |
| Um `pip install paddleocr` falha hoje (1.3) e a mensagem não é óbvia | Se precisar do laboratório Python, `uv venv --python 3.12` desde o primeiro comando |
| A GPU é compartilhada com outras cargas; `--accel auto` pode disputar VRAM no meio de um jogo ou treino | `--accel cpu` no atalho interativo e `auto`/`cuda` só no lote (2.9). O daemon deve tolerar `cudaMalloc` falhando e cair para CPU nessa chamada, não abortar |

---

## 5. Verificação

**Automatizável em CI/host (`cargo test`):**
- [ ] `ocr-core` devolve `吾輩は猫である。名前はまだ無い。` para `ja_h.png` — o mesmo
      arquivo em que o Tesseract devolveu `EEIIETH D, BEITEIEEL,` (1.4). O teste
      guarda a premissa central do relatório: que trocar de motor resolve CJK.
- [ ] `ocr-core` devolve `안녕하세요 세계` para `ko.png` com o modelo `korean`, e algo
      diferente disso com o modelo japonês — guarda que a seleção de modelo tem efeito
      real, e não é decorativa.
- [ ] Uma linha em português com diacríticos (`ação`, `você`, `pêssego`) sai íntegra
      pelo modelo `latin` — guarda o caso de uso doméstico, que é o mais frequente e
      o mais fácil de esquecer de testar.
- [ ] A confiança devolvida para uma imagem de ruído puro fica abaixo do limiar —
      guarda que o *fallback* de 2.4 tem sinal para operar.
- [ ] Agregação temporal: dada uma sequência sintética de leituras
      `["A","A","A","B","B"]`, o resultado tem exatamente 2 legendas com intervalos
      corretos — guarda o estágio 4 de 2.6, que é lógica pura e a maior fonte de bugs
      silenciosos do pipeline de vídeo.

**Precisa de GPU e do `cudnn` instalado (declarar como não verificado até rodar):**
- [ ] Com `--accel auto` e `cudnn` ausente, o daemon **cai para CPU e diz isso no
      log** — não trava e não fica em silêncio. Guarda o *fallback* de 2.9.
- [ ] Com `--accel cuda` e `cudnn` ausente, o processo **falha alto** na
      inicialização, sem cair para CPU. Guarda que o modo "exigir GPU" é honesto.
- [ ] Com `--accel auto` e o *stack* completo, o *log* registra `CUDAExecutionProvider`
      ativo, e `--accel cpu` no mesmo *host* registra CPU — guarda que a *flag* manda.
- [ ] O ganho de latência CPU × CUDA é medido, não presumido (2.9). Decide o
      comportamento do `auto` no caminho interativo: abaixo de ~2×, o atalho fica
      sempre em CPU e a GPU serve só aos fluxos em lote.

**Só no uso real, manualmente:**
- [ ] `SUPER+SHIFT+T` → seleção → texto no clipboard em menos de 1 s, com o daemon
      quente.
- [ ] Uma página de mangá real: comparar PP-OCRv5 com `manga-ocr` **antes** de gastar
      os 2–3 d da F5. Se o PP-OCRv5 acertar, a F5 não existe.

---

## 6. Riscos

1. **O caminho Rust é sólido, mas apoiado num crate jovem.** `oar-ocr` tem 163
   estrelas e 0,9.2 como versão. Se ele for abandonado, o custo de saída não é
   catastrófico — os modelos ONNX continuam válidos e `ort` continua mantido —, mas
   você herda o pipeline: pré-processamento, decodificação CTC e agrupamento de
   caixas, algo entre 3 e 5 dias-dev [modelado]. Mantenha `ocr-core` como fronteira
   para que essa substituição seja local.
2. **`ort` nunca chegou ao 1.0 estável.** Está em *release candidate* desde antes de
   2025 e continua em `2.0.0-rc.13`. É amplamente usado — 6,68 M de *downloads* em 90
   dias —, mas quebras de API entre *rc* são esperadas, e vão custar tempo em cada
   atualização.
3. **A F5 (mangá) pode consumir um quarto do cronograma para atender um nicho.** É a
   única fase que exige implementar um laço de inferência autorregressivo à mão. Se o
   objetivo real é "ler mangá em japonês", considere seriamente usar o **koharu**
   pronto e restringir o projeto próprio ao resto — a decisão de construir precisa ser
   consciente, não inercial.
4. **A qualidade em japonês do PP-OCRv5 é boa, não excelente.** A própria *model card*
   publica 0,7577 no cenário japonês. Isso significa que cerca de uma linha em cada quatro
   sai com algum erro na avaliação deles, que marca a linha inteira como errada se
   qualquer caractere — pontuação inclusive — divergir. Para consulta e tradução é
   ótimo; para transcrição fiel de documento, não é.
5. **O escopo "imagens e vídeos, multilíngue, uso diário" é três produtos.** Atalho de
   captura, leitor de arquivos e extrator de legenda compartilham o núcleo e nada
   mais — interfaces, ergonomia e critérios de erro são diferentes. Entregar F0–F2 e
   usar por duas semanas antes de decidir F5/F6/F7 vale mais do que qualquer
   estimativa deste relatório.

---

## 7. Arquivos tocados

Todos **novos** — o projeto está vazio (1.1).

| Arquivo | Mudança |
|---|---|
| `Cargo.toml` (workspace) | **novo** — membros `ocr-core`, `ocrd`, `ocrd-client`, `ocr-video` |
| `crates/ocr-core/src/lib.rs` | **novo** — API `recognize()`, fronteira de isolamento do `oar-ocr` |
| `crates/ocr-core/src/models.rs` | **novo** — registro dos 4 modelos, *download* e cache |
| `crates/ocr-core/src/lang.rs` | **novo** — *fallback* por confiança (2.4) |
| `crates/ocr-core/src/accel.rs` | **novo** — sondagem de GPU, seleção de *execution provider*, *flag* `--accel` (2.9) |
| `crates/ocrd/src/main.rs` | **novo** — daemon, *socket* Unix, modelos residentes |
| `crates/ocrd-client/src/main.rs` | **novo** — cliente CLI, stdin → stdout |
| `crates/ocr-video/src/lib.rs` | **novo** — `ffmpeg`, dedup SSIM, agregação temporal, `.srt` (F6) |
| `crates/ocr-manga/src/lib.rs` | **novo** — laço de decodificação do `manga-ocr` (F5, opcional) |
| `tests/data/{en,ja_h,zh,ko}.png` | **novo** — as quatro imagens já geradas e medidas em 1.4 |
| `tests/data/pt_diacriticos.png` | **novo** — falta gerar |
| `contrib/hyprland.conf.example` | **novo** — o *bind* de 2.7 |
| `contrib/ocrd.service` | **novo** — unidade *systemd --user* |
| `README.md` | **novo** — inclusive o `pacman -S cudnn` como pré-requisito |

---

## 8. O que NÃO foi verificado

- **Nenhuma linha de Rust foi escrita ou compilada.** `oar-ocr`, `ort` e os modelos
  PP-OCRv5 nunca rodaram nesta máquina. Toda a seção 2.3 apoia-se em documentação
  publicada e em métricas de repositório, não em execução.
- **A acurácia do PP-OCRv5 nas imagens de teste é desconhecida.** A tabela de 1.4 mede
  o Tesseract, não o substituto proposto. A afirmação "trocar de motor resolve CJK" é
  uma hipótese fundamentada, e é exatamente o que a F0 existe para confirmar ou
  derrubar em meio dia.
- **Se o dicionário do `pp-ocrv5_mobile_rec` cobre diacríticos do português** não foi
  verificado — é por isso que `latin_pp-ocrv5_mobile_rec` está no plano e que existe
  um teste dedicado na seção 5.
- **O EP CUDA nunca foi carregado aqui.** `cudnn` não está instalado (1.3), e a
  compatibilidade CUDA 13.3 ↔ ORT ≥ 1.27 vem da tabela oficial do ONNX Runtime, não
  de um teste local.
- **O ganho GPU × CPU é [modelado].** O "3–10×" de 2.9 vale para o lote e vem da
  literatura de inferência ONNX, não de medição nos modelos *mobile* nesta máquina.
  Para uma linha de tela isolada, o ganho pode ser próximo de zero — a F0 mede isso.
- **A afirmação sobre o `Windows.Media.Ocr` (2.8) apoia-se na documentação da
  Microsoft, não em teste.** Não rodei o PowerToys nem o motor do Windows; a
  comparação de arquitetura ("pipeline clássico, não transformer") é inferida do
  tamanho dos *language packs* e do comportamento, e a Microsoft não publica os
  internos.
- **Todos os números de esforço são [modelado].** Não há histórico deste projeto para
  calibrá-los.
- **A latência do daemon quente não foi medida** — o alvo de "menos de 1 s" em 5 vem
  do orçamento estabelecido em 1.4, não de uma medição do sistema proposto.
- **Hipótese refutada, registrada para não voltar:** *"não usar Python obriga a abrir
  mão dos bons modelos de OCR"* — **é falso**, e está refutado em 2.1 e 2.4. Os
  modelos PP-OCRv5/v6 são arquivos ONNX publicados com tamanho conhecido, e o koharu
  (5.495 estrelas) prova em produção que a orquestração em Rust funciona. O que Python
  retém é o treinamento e a exportação, nenhum dos dois necessário aqui.
- **Hipótese refutada, registrada para não voltar:** *"o Tesseract já está instalado,
  então é o caminho barato"* — **é falso**. Ele não tem os idiomas (1.3), erra em
  inglês impresso limpo (1.4), tem falha documentada em japonês vertical, e o binding
  Rust mais usado (`leptess`) não recebe *release* desde fevereiro de 2023.

---

> Nenhum item deste relatório foi executado no aplicativo proposto, porque ele não
> existe: o diretório do projeto está vazio. As únicas medições reais aqui são as de
> ambiente (seção 1.2), as de ausência (1.3) e a tabela do Tesseract em 1.4, todas
> obtidas por execução direta nesta máquina em 06/09/2026. Tudo o mais — acurácia do
> PP-OCRv5, latência do daemon, esforço das fases — está listado na seção 8 como não
> verificado, e a fase F0 existe para converter a maior parte disso em número medido
> em meio dia de trabalho.
