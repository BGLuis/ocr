# Arch User Repository (AUR) — `ocr`

Este diretório contém os arquivos de empacotamento para publicação do `ocr` no [AUR (Arch User Repository)](https://aur.archlinux.org/), permitindo a instalação via gerenciadores como `yay` e `paru`.

## 📦 Instalação via `yay` ou `paru`

Uma vez publicado no AUR:

```sh
# Usando yay
yay -S ocr

# Usando paru
paru -S ocr
```

Após instalar, ative o serviço de usuário do daemon do systemd:

```sh
systemctl --user daemon-reload
systemctl --user enable --now ocrd
```

## 🛠️ Instalação manual local (sem AUR)

Para compilar e instalar localmente a partir deste repositório:

```sh
# Compilar e instalar binários
cargo install --locked --path crates/ocrd
cargo install --locked --path crates/ocrd-client
install -Dm755 contrib/ocr ~/.local/bin/ocr

# Ou para testar a criação do pacote makepkg localmente gerando o tarball da versão:
tar -czf "aur/ocr-0.1.0.tar.gz" --transform "s|^.|ocr-0.1.0|" --exclude="./target" --exclude="./.git" .
cd aur
makepkg -si
```

## 🚀 Publicação no AUR

O repositório inclui a workflow `.github/workflows/release.yml` que:
1. Constrói e testa os binários de release.
2. Cria uma GitHub Release com a tag `v*.*.*`.
3. Atualiza automaticamente o `PKGBUILD` com o checksum SHA256 do tarball da release.
4. Gera o `.SRCINFO` atualizado via `makepkg --printsrcinfo`.
5. Publica os arquivos no repositório AUR (`ssh://aur@aur.archlinux.org/ocr.git`).

### Configuração necessária no GitHub

Para que o GitHub Actions consiga autenticar e enviar os commits para o AUR:

1. **Conta no AUR**: Crie uma conta em [aur.archlinux.org](https://aur.archlinux.org/).
2. **Chave SSH**:
   ```sh
   ssh-keygen -t ed25519 -C "aur-deploy" -f ~/.ssh/aur_deploy
   ```
3. **Chave pública**: Copie o conteúdo de `~/.ssh/aur_deploy.pub` e adicione no seu perfil do AUR (*My Account* -> *SSH Public Key*).
4. **Secret no GitHub**: Adicione o conteúdo da chave privada `~/.ssh/aur_deploy` nos segredos do repositório GitHub (*Settings* -> *Secrets and variables* -> *Actions* -> *Repository secrets*) com o nome:
   `AUR_SSH_PRIVATE_KEY`
