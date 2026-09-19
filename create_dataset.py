import pandas as pd

data = {
    'v1': ['ham', 'spam', 'ham', 'spam', 'ham'],
    'v2': [
        'Hey, are we still meeting today?',
        'WINNER! You have won a $1000 gift card. Click here now!',
        'Can you please send me the project report?',
        'URGENT! Your account balance is low. Claim free bonus.',
        'Ok, I will call you later.'
    ]
}

# CSV फाइल तयार करा
df = pd.DataFrame(data)
df.to_csv('spam.csv', index=False)
print("✅ dummy spam.csv")