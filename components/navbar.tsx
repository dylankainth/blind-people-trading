import React from "react";
import { NavigationMenu, NavigationMenuList, NavigationMenuLink, navigationMenuTriggerStyle } from "@/components/ui/navigation-menu";
import Link from "next/link";

const Navbar: React.FC = () => {
    return (
        <nav className="flex items-center justify-between p-4 bg-white-800 text-black shadow-md">
            <div className="text-lg font-bold">MyApp</div>
            <NavigationMenu>
                <NavigationMenuList>
                    <NavigationMenuLink asChild>
                        <Link href="/docs" className={navigationMenuTriggerStyle()}>
                            klnreslkrnesjirs
                        </Link>
                    </NavigationMenuLink>
                </NavigationMenuList>
            </NavigationMenu>
        </nav>
    );
};


export default Navbar;