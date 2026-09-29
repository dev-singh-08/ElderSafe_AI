document.addEventListener("DOMContentLoaded", function () {

    // Automatically hide flash messages
    const messages = document.querySelectorAll(".flash");

    messages.forEach(function (message) {

        setTimeout(function () {

            message.style.opacity = "0";
            message.style.transform = "translateY(-5px)";

            setTimeout(function () {
                message.remove();
            }, 300);

        }, 4000);

    });


    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(function (link) {

        link.addEventListener("click", function (event) {

            const target = document.querySelector(
                this.getAttribute("href")
            );

            if (target) {

                event.preventDefault();

                target.scrollIntoView({
                    behavior: "smooth"
                });

            }

        });

    });

});