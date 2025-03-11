from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import random

load_dotenv()
# agent = Agent(
#     model='openai:gpt-4o',
#     system_prompt='You are z helpful assistant. Be concise and reply with one sentence.'
# )

# result = agent.run_sync('Where dies "Hello World" come from')
# print(result.data)

die_game_agent = Agent(
    model='openai:gpt-4o-mini',
    deps_type=str,
    system_prompt=(
        "You're a dice game, You should roll the die and see the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response"
    ),
)

@die_game_agent.tool_plain
def rool_die() -> str:
    """ Roll a six-sided die and return the result."""
    num = random.randint(1,6)
    print("rolled: ", num)
    return str(num)

@die_game_agent.tool
def get_player_name(ctx:RunContext[str])->str:
    """ Get the player name. """
    return ctx.deps

dice_result = die_game_agent.run_sync('My guess is 3', deps='Amadou')
print(dice_result.data)
print(dice_result.all_messages())