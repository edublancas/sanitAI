# SanitAI

A secure middleware that sanitizes OpenAI API traffic by:

- Automatically detecting and removing Personal Identifiable Information (PII)
- Preserving message context and meaning
- Operating as a drop-in reverse proxy for existing OpenAI integrations

What your user sends:

> Hello, my card number is 4111-1111-1111-1111. Call me at (123) 456-7890

What the OpenAI API sees:

> Hello, my card number is `<VISA-CARD>`. Call me at `<US-NUMBER>`

## Getting started


```sh
# get the code
git clone https://github.com/edublancas/sanitAI
cd sanitAI/

# set your OpenAI key (used for the agent that helps you define the PII rules)
export OPENAI_API_KEY=SOME-OPENAI-KEY

# build the docker image
docker build -t presidioui .

# run
docker run -p 5001:80 --rm --name presidioui \
-e OPENAI_API_KEY=$OPENAI_API_KEY \
  presidioui
```

Then, open: `http://localhost:5001/admin/`

Login:

- Email: admin@example.com
- Password: admin123


Run a sample script that makes an API call, open a new terminal and execute:

```sh
export OPENAI_API_KEY=SOME-OPENAI-KEY
pip install openai

python example.py
```

Then, look at the logs from the server and you'll see that the user sent:

> Hello, my card number is 4111-1111-1111-1111. Call me at (123) 456-7890

But the reverse proxy intercepts the request and sends this to OpenAI:

> Hello, my card number is `<VISA-CARD>`. Call me at `<US-NUMBER>`
