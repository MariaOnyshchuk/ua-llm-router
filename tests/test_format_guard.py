from router.format_guard import apply, detect


def test_detects_the_four_constraints_from_the_request_text():
    assert detect("Напиши загадку, але не використовуй жодних ком.") == ["no_comma"]
    assert detect("Обгорніть всю вашу відповідь подвійними лапками.") == ["quotes"]
    assert detect("Відповідь має бути англійською мовою і всіма малими літерами.") == ["lower"]
    assert detect("Переконайтеся, що у всій відповіді використовуються лише великі літери.") == ["upper"]
    assert detect("Вона має бути англійською мовою і без великих літер.") == ["lower"]


def test_does_not_fire_on_counts_or_on_requests_that_want_commas():
    assert detect("Вірш має містити щонайменше 3 слова, написані великими літерами.") == []
    assert detect("Поверни список через кому без пробілів після ком: три кольори.") == []
    assert detect("Заголовок візьміть у подвійні кутові дужки, тобто <<заголовок>>.") == []
    assert detect("Що подарувати татові на 60-річчя?") == []


def test_apply_repairs_and_reports_what_it_changed():
    fixed, made = apply("Не використовуй жодних ком і обгорни відповідь подвійними лапками.", "Привіт, світе, я тут")
    assert fixed == '"Привіт світе я тут"' and made == ["no_comma", "quotes"]
    fixed, made = apply("Write it in English, all lowercase.", "Hello World")
    assert fixed == "hello world" and made == ["lower"]
    fixed, made = apply("Обгорніть всю відповідь подвійними лапками.", '"Уже в лапках"')
    assert fixed == '"Уже в лапках"' and made == []
    fixed, made = apply("Що таке фотосинтез?", "Процес, який …")
    assert fixed == "Процес, який …" and made == []
