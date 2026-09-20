Run it in a second Terminal window—without quotation marks—while `edge0 serve` remains running in the first:

```bash
edge0ask "Explain how an emergency-department forecasting model could detect distribution drift"
```

The quoted portion keeps the entire prompt together as one argument. After a short delay, Edge0’s response should print directly in Terminal.

If you see:

```text
zsh: command not found: edge0ask
```

reload the function:

```bash
source ~/.zshrc
```

If you see:

```text
jq: command not found
```

install it:

```bash
brew install jq
```

You can then ask anything similarly:

```bash
edge0ask "Write a SQL Server 2014 query to detect changes in hourly ED arrival patterns"
```

```bash
edge0ask "Design a clinical validation plan for an emergency-department forecasting model"
```

