import streamlit as st


def attngan_explanation():
    st.markdown(
        """Text-to-image conversion is a very interesting application of the neural networks (Generative Adverserial Networks to be specific). Automatically generating images according to natural
        language descriptions is a fundamental problem in many
        applications, such as art generation and computer-aided design. It also drives research progress in **multimodal learning**
        and inference across vision and language, which is one of
        the most active research areas in recent years.  \n\n One of such architecture which used the **attention mechanism** (borrowed from NLP) was introduced in 2018 by collaboration of Lehigh University, Microsoft Research, Rutgers University, Duke University.  \n\nBefore we dive into the specifics of this research let's first understand what we are dealing with here. The problem is Text-to-Image conversion. 
        So, you give the textual description of a particular object as a description and you will get an image as an output (usually RGB).
""", unsafe_allow_html=True)

    st.markdown("#")
    st.image("img/img-to-text.png", caption="A dog and a kid are playing frisbee")

    st.markdown("#")
    st.markdown(
        """
        When we read the description, we as humans are naturally gifted with this capability of imagination, we are able to visualize this scene without ever actually having to see that precise scene. So that's what we want our models to do: that is, given a description the algorithm should be able to generate an image.

        """
    )

    st.markdown("---")

    st.header("Getting into the specifics of AttnGAN architecture")

    st.subheader("Architecture - Motivation")
    st.markdown("""
    **1. Encoder-Decoder -> Sequence-to-Sequence**

    Image to text conversion looks like a encoder decoder problem. You encode the image into a single vector and then decode it into it's corresponding text. But the researchers (of AttnGAN) turned it into a sequence to sequence problem. The sequential processing is not a problem for the text part because RNNs/LSTMs do sequentially process the text but the problem lies in the sequential processing of the image part, which was first seen in StackGAN.

    **2. Image generation as an m-stage process**

    The second motivation was making the image generation as an m-stage process. So, you would have m different generators and their corresponding discriminators to generate a final image.

    **3. Use advances in Seq2Seq (Attention)**

    The main motivation to treat this problem as Seq2Seq problem was to leverage the attention mechanism, which is a concept borrowed from the recent advancements in Natural Language Processing.

    **4. Make text and image agree to each other**

    The researchers also made sure that the text and image were aligned to each other. They included a component in the final layer's loss, which we will talk about in detail further.
    """)