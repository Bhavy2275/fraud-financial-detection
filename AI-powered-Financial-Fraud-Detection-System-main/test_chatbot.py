from chatbot import AtlasBot

def test_chatbot_responses():
    bot = AtlasBot()
    
    # Test basic queries
    response = bot.get_response("hi")
    assert any(keyword in response for keyword in ["Hello", "Hi", "Welcome"])
    
    response = bot.get_response("help me")
    assert any(keyword in response for keyword in ["help", "assist"])
    
    response = bot.get_response("features")
    assert any(keyword in response for keyword in ["99.9%", "detection"])
    
    # Test unknown queries
    response = bot.get_response("random text")
    assert response in bot.responses["default"]

if __name__ == "__main__":
    test_chatbot_responses()
    print("All tests passed!") 