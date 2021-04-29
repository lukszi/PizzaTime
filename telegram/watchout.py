import json

import requests


with open("res/config.json", "r", encoding="UTF-8") as config_file:
    config = json.load(config_file)
    bot_token = config["bot_token"]

CHAT_ID_FILE_PATH = "res/chat_ids.json"
with open(CHAT_ID_FILE_PATH, "r", encoding="UTF-8") as chat_id_file:
    chat_ids = json.load(chat_id_file)


def telegram_bot_sendtext(message, chat_id):
    """
    Send message to specified chat

    :param message: message to be sent
    :param chat_id: id of the chat
    :return: API response data
    """
    send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?' \
                f'chat_id={chat_id}' \
                f'&parse_mode=Markdown' \
                f'&text={message}'

    response = requests.get(send_text)

    return response.json()


def warn(n=10):
    """
    Sends out a warning to all open chatIDs
    :param n: number of warnings to be sent, default is 10
    :return:
    """
    for i in range(0, n):
        for chat_id in chat_ids:
            telegram_bot_sendtext("LAUF", chat_id)


def get_new_chat_ids():
    """
    Register new chat ids
    """
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    response = requests.get(url)
    updates = response.json()
    print(f"updates: {updates}")

    added_new_id_to_list = False
    for result in updates["result"]:
        chat_id = result["message"]["from"]["id"]
        if chat_id in chat_ids:
            continue
        added_new_id_to_list = True
        chat_ids.append(chat_id)

    if added_new_id_to_list:
        with open(CHAT_ID_FILE_PATH, "w", encoding="UTF-8") as chat_id_file:
            json.dump(chat_ids, chat_id_file)


if __name__ == '__main__':

    get_new_chat_ids()

    for cht_id in chat_ids:
        test = telegram_bot_sendtext("Testing Telegram bot", cht_id)
        print(test)
