import streamlit as st


def attngan_explanation():
    st.markdown(
        """Text-to-image conversion is a very interesting application of the neural networks (Generative Adverserial Networks to be specific). Automatically generating images according to natural
        language descriptions is a fundamental problem in many
        applications, such as art generation and computer-aided design. It also drives research progress in **multimodal learning**
        and inference across vision and language, which is one of
        the most active research areas in recent years.  \n\n One of such architecture which used the **attention mechanism** (borrowed from NLP) was introduced in 2018 by collaboration of Lehigh University, Microsoft Research, Rutgers University, Duke University.  \n\nBefore we dive into the specifics of this research let's first understand what we are dealing with here. The problem is Text-to-Image conversion. 
        So, you give the textual description of a particular object as a description and you will get an image as an output (usually RGB).
""",
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
    **1. Encoder-Decoder -> Sequence-to-Sequence**

    Image to text conversion looks like a encoder decoder problem. You encode the image into a single vector and then decode it into it's corresponding text. But the researchers (of AttnGAN) turned it into a sequence to sequence problem. The sequential processing is not a problem for the text part because RNNs/LSTMs do sequentially process the text but the problem lies in the sequential processing of the image part, which was first seen in StackGAN.

    **2. Image generation as an m-stage process**

    The second motivation was making the image generation as an m-stage process. So, you would have m different generators and their corresponding discriminators to generate a final image.

    **3. Use advances in Seq2Seq (Attention)**

    The main motivation to treat this problem as Seq2Seq problem was to leverage the attention mechanism, which is a concept borrowed from the recent advancements in Natural Language Processing.

    **4. Make text and image agree to each other**

    The researchers also made sure that the text and image were aligned to each other. They included a component in the final layer's loss, which we will talk about in detail further.
    """
    )

    st.markdown("#")

    st.header("Architecture")
    st.markdown("#")
    st.image("img/architecture.png", caption="AttnGAN architecture")
    st.markdown("#")

    st.markdown(
        """
    This architecture may look little compilcated at the first glance. So, let's look into it part-by-part.
    """
    )

    st.subheader("Text Encoder")
    st.markdown("#")
    st.image("img/text-encoder.png", caption="Text Encoder")

    st.markdown(
        """
    In the text encoder part, we have a bidirectional LSTM. We concantenate hidden state for forward and backward directions. By doing this, we get a single hidden state per time stamp.

    We have two features off the bidirectional RNN:

    1) **Sentence feature**: Final hidden state, which is a d dimensional vector

    2) **Word features**: Hidden states from all timesteps, which is a D x T dimensional matrix (where T is the number of words)
    """
    )

    st.subheader("Conditioning Augmentation")
    st.markdown(
        """
    Conditioning augmentation is the next step, in which the latent variables are sampled randomly from the Gaussian distribution
    """
    )
    st.markdown("#")
    st.image("img/noise-vector.png", caption="noise vector")
    st.markdown(
        """
        The sentence feature which is the output of the B-RNN is split into u and sigma which is done using fully connected layer. The first half of that is used as u and second half as sigma. We combine the u and sigma with the noise vector. The noise vector is added to have a higher variation in generate images even for the same caption due to F^ca.
    """
    )

    st.markdown("This blog post is still under development...")
