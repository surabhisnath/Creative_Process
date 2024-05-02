from together_call import main_together
from gpt_call import main_gpt
from anthropic_call import main_anthropic
from palm_call import main_palm
from gemini_call import main_gemini

if __name__ == "__main__":
    main_together()
    main_gpt()
    main_anthropic()
    main_palm()
    main_gemini()