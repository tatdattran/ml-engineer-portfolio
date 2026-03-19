const form = document.getElementById("predictForm")
const summary = document.getElementById("error-summary")
const loading = document.getElementById("loading")
const resultBox = document.getElementById("result")
const button = form.querySelector("button")

form.onsubmit = async function(e){
    e.preventDefault()

    let isValid = true
    let errors = []

    const groups = document.querySelectorAll(".form-group")

    summary.style.display = "none"
    summary.innerHTML = ""

    groups.forEach(group => {
        const input = group.querySelector("input, select")
        const errorMsg = group.querySelector(".error-message")

        if (!input.value || input.value <= 0) {
            input.classList.add("error")
            if (errorMsg) errorMsg.style.display = "block"
            errors.push(input.name)
            isValid = false
        } else {
            input.classList.remove("error")
            if (errorMsg) errorMsg.style.display = "none"
        }
    })

    if (!isValid) {
        summary.innerHTML = "⚠️ Fix: <br>" + errors.join("<br>")
        summary.style.display = "block"

        form.classList.add("shake")
        setTimeout(() => form.classList.remove("shake"), 300)
        return
    }

    // LOADING
    loading.style.display = "block"
    button.disabled = true

    const data = Object.fromEntries(new FormData(form).entries())

    for(let k in data){
        if(!isNaN(data[k])) data[k] = Number(data[k])
    }

    const res = await fetch("/predict", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(data)
    })

    const result = await res.json()

    loading.style.display = "none"
    button.disabled = false

    resultBox.innerText = "$" + Number(result.predicted_price).toLocaleString()
}

// AUTO FIX ERROR
document.querySelectorAll("input, select").forEach(el => {
    el.addEventListener("change", () => {
        el.classList.remove("error")
        const group = el.closest(".form-group")
        const msg = group.querySelector(".error-message")
        if (msg) msg.style.display = "none"
    })
})
