import logging
import setup_common
import sys

errors = 0  # Определение переменной 'errors'
log = logging.getLogger('sd')

# ANSI escape-код для желтого цвета
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'


def install_tensorflow():
    setup_common.check_repo_version()
    setup_common.check_python()

    # Обновление pip, если необходимо
    setup_common.install('--upgrade pip')

    setup_common.install_requirements('requirements_annex.txt', check_no_verify_flag=True)


def main_menu():
    setup_common.clear_screen()
    while True:
        print('\n Установка и настройка проекта:\n')
        print('1. Установка виртуальной среды *venv*')
        print('2. Выход')

        choice = input('\nСделайте выбор: ')
        print('')

        if choice == '1':
            install_tensorflow()
        elif choice == '2':
            print('Выход из меню')
            sys.exit()
        else:
            print('Выберите между 1-2')


if __name__ == '__main__':
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    main_menu()
